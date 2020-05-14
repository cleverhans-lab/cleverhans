# pylint: disable=missing-docstring
import logging

import numpy as np
import torch as ch

from cleverhans.future.torch.utils import get_or_guess_labels
from cleverhans import utils


_logger = utils.create_logger(
    "cleverhans.future.torch.attacks.carlini_wagner_l2")
_logger.setLevel(logging.INFO)


def carlini_wagner_l2(model_fn, x, y=None, targeted=False, confidence=0,
                      learning_rate=5e-3, binary_search_steps=5,
                      max_iterations=1000, abort_early=True,
                      initial_const=1e-2, clip_min=0, clip_max=1,
                      logging=True):
  """
  This attack was originally proposed by Carlini and Wagner. It is an
  iterative attack that finds adversarial examples on many defenses that
  are robust to other attacks.
  Paper link: https://arxiv.org/abs/1608.04644

  At a high level, this attack is an iterative attack using Adam and
  a specially-chosen loss function to find adversarial examples with
  lower distortion than other attacks. This comes at the cost of speed,
  as this attack is often much slower than others.

  :param model_fn: a callable that takes an input tensor and returns
            the model logits.
  :param x: input tensor.
  :param y: (optional) Tensor with true labels. If targeted is true,
            then provide the target label. Otherwise, only provide this
            parameter if you'd like to use true labels when crafting
            adversarial samples. Otherwise, model predictions are used
            as labels to avoid label leaking. Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being
            more like y.
  :param confidence: Confidence of adversarial examples: higher produces
            examples with larger l2 distortion, but more
            strongly classified as adversarial.
  :param learning_rate: The learning rate for the attack algorithm.
            Smaller values produce better results but are
            slower to converge.
  :param binary_search_steps: The number of times we perform binary
            search to find the optimal tradeoff-
            constant between norm of the purturbation
            and confidence of the classification.
  :param max_iterations: The maximum number of iterations. Setting this
            to a larger value will produce lower distortion
            results. Using only a few iterations requires
            a larger learning rate, and will produce larger
            distortion results.
  :param abort_early: If true, allows early aborts if gradient descent
            is unable to make progress (i.e., gets stuck in
            a local minimum).
  :param initial_const: The initial tradeoff-constant to use to tune the
            relative importance of size of the perturbation
            and confidence of classification.
            If binary_search_steps is large, the initial
            constant is not important. A smaller value of
            this constant gives lower distortion results.
  :param clip_min: (optional float) Minimum input component value
  :param clip_max: (optional float) Maximum input component value
  :param logging: If true, logs useful messages in logger.
  """
  # Keep everything on same device
  device = x.device
  if y is None:
    y = ch.argmax(model_fn(x), 1)

  # Get number of classes
  nb_classes = model_fn(x[:1]).shape[1]
  batch_size = x.shape[0]

  repeat = binary_search_steps >= 10

  y_onehot = ch.nn.functional.one_hot(y, nb_classes) * 1.0

  # the variable we're going to optimize over
  modifier = ch.zeros_like(x, requires_grad=True, device=device)

  # set the lower and upper bounds accordingly
  lower_bound = ch.zeros(batch_size, device=device)
  upper_bound = ch.ones(batch_size,  device=device) * 1e10
  const = ch.ones(batch_size,  device=device) * initial_const

  # placeholders for the best l2, score, and instance attack found so far
  o_bestl2 = ch.ones(batch_size, device=device) * np.inf
  o_bestscore = ch.ones(batch_size, device=device) * -1

  x = ch.clamp(x, clip_min, clip_max)
  orig_x = x.clone().detach()
  o_bestattack = x.clone().detach()

  # re-scale instances to be within range [0, 1]
  x = ch.clamp((x - clip_min) / (clip_max - clip_min), 0, 1)
  # now convert to [-1, 1]
  x = (x * 2) - 1
  # convert to tanh-space
  x = arctanh(x * .999999)

  optimizer = ch.optim.Adam([modifier], lr=learning_rate)
  for outer_step in range(binary_search_steps):
    if logging:
      _logger.debug("  Binary search step %s of %s",
                    outer_step, binary_search_steps)

    # The last iteration (if we run many steps) repeat the search once.
    if repeat and outer_step == binary_search_steps - 1:
      const = upper_bound

    bestl2 = ch.ones(batch_size, device=device) * np.inf
    bestscore = ch.ones(batch_size, device=device) * -1
    prev_b_loss = 1e6

    for iteration in range(max_iterations):
      b_loss, l2s, scores, nimg = optimization_step(x, orig_x, model_fn,
                                                    modifier,  optimizer,
                                                    targeted, y_onehot, const,
                                                    clip_min, clip_max,
                                                    confidence)

      if logging and iteration % ((max_iterations // 10) or 1) == 0:
        _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                       "l2={:.3g} f={:.3g}").format(
            iteration, max_iterations, b_loss,
            ch.mean(l2s), ch.mean(scores)))

      # check if we should abort search if we're getting nowhere.
      if abort_early and iteration % ((max_iterations // 10) or 1) == 0:
        if b_loss > prev_b_loss * .9999:
          if logging:
            _logger.debug("    Failed to make progress; stop early")
          break
        prev_b_loss = b_loss

      score_preds = ch.argmax(scores, 1)
      adjusted_score_preds = ch.argmax(
          adjust_confidence(scores, y, confidence, targeted), 1)

      # adjust the best result found so far
      for e, (l2, ii) in enumerate(zip(l2s, nimg)):
        if compare(adjusted_score_preds[e], y[e], targeted):
          if l2 < bestl2[e]:
            bestl2[e] = l2
            bestscore[e] = score_preds[e]
          if l2 < o_bestl2[e]:
            o_bestl2[e] = l2
            o_bestscore[e] = score_preds[e]
            o_bestattack[e] = ii

    # adjust the constant as needed
    for e in range(batch_size):
      if compare(bestscore[e], y[e], targeted) and bestscore[e] != -1:
        # success, divide const by two
        upper_bound[e] = min(upper_bound[e], const[e])
        if upper_bound[e] < 1e9:
          const[e] = (lower_bound[e] + upper_bound[e]) / 2
      else:
        # failure, either multiply by 10 if no solution found yet
        #          or do binary search with the known upper bound
        lower_bound[e] = max(lower_bound[e], const[e])
        if upper_bound[e] < 1e9:
          const[e] = (lower_bound[e] + upper_bound[e]) / 2
        else:
          const[e] *= 10

    if logging:
      _logger.debug("  Successfully generated adversarial examples " +
                    "on {} of {} instances.".format(
                        ch.sum(upper_bound < 1e9), batch_size))

    if logging:
      mean = ch.mean(ch.sqrt(o_bestl2[o_bestl2 < 1e9]))
      _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

  return o_bestattack.detach()


def arctanh(x):
  return 0.5 * ch.log((1 + x) / (1 - x))


def adjust_confidence(x, y, confidence, targeted):
  x_ = x.clone().detach()
  for i in range(x_.shape[0]):
    if targeted:
      x_[i, y[i]] -= confidence
    else:
      x_[i, y[i]] += confidence
  return x_


def compare(x, y, targeted):
  if targeted:
    return x == y
  else:
    return x != y


def optimization_step(x, orig_x, model_fn, modifier, optimizer, targeted,
                      y_onehot, const, clip_min, clip_max, confidence):
  optimizer.zero_grad()

  newimg = (ch.tanh(modifier + x) + 1) / 2
  newimg = newimg * (clip_max - clip_min) + clip_min

  loss2 = ch.pow(newimg - orig_x, 2)
  loss2 = ch.sum(loss2.view(loss2.size(0), -1), 1)

  output = model_fn(newimg)
  real = ch.sum(y_onehot * output, 1)
  other, _ = ch.max(((1 - y_onehot) * output - y_onehot * 1e4), 1)

  if targeted:
    loss1 = ch.clamp(other - real + confidence, min=0)
  else:
    loss1 = ch.clamp(real - other + confidence, min=0)

  loss = ch.sum(loss2 + const * loss1)

  loss.backward()
  optimizer.step()

  return loss.item(), loss2, output, newimg
