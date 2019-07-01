"""The Fast Gradient Method attack."""

import numpy as np
import tensorflow as tf


def fast_gradient_method(model_fn, x, eps, norm, clip_min=None, clip_max=None, y=None,
                         targeted=False, sanity_checks=False):
  """
  Tensorflow 2.0 implementation of the Fast Gradient Method.
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError("Norm order must be either np.inf, 1, or 2.")

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(tf.math.greater_equal(x, clip_min))

  if clip_max is not None:
    asserts.append(tf.math.less_equal(x, clip_max))

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    y = tf.argmax(model_fn(x), 1)

  grad = compute_gradient(model_fn, x, y, targeted)

  optimal_perturbation = optimize_linear(grad, eps, norm)
  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x


# Due to performance reasons, this function is wrapped inside of tf.function decorator.
# Not using the decorator here, or letting the user wrap the attack in tf.function is way
# slower on Tensorflow 2.0.0-alpha0.
@tf.function
def compute_gradient(model_fn, x, y, targeted):
  """
  Computes the gradient of the loss with respect to the input tensor.
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor
  :param y: Tensor with true labels. If targeted is true, then provide the target label.
  :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                    try to make the label incorrect. Targeted will instead try to move in the
                    direction of being more like y.
  :return: A tensor containing the gradient of the loss with respect to the input tensor.
  """
  loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits
  with tf.GradientTape() as g:
    g.watch(x)
    # Compute loss
    loss = loss_fn(labels=y, logits=model_fn(x))
    if targeted:  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
      loss = -loss

  # Define gradient of loss wrt input
  grad = g.gradient(loss, x)
  return grad


def optimize_linear(grad, eps, norm=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

  :param grad: tf tensor containing a batch of gradients
  :param eps: float scalar specifying size of constraint region
  :param norm: int specifying order of norm
  :returns:
    tf tensor containing optimal perturbation
  """

  # Convert the iterator returned by `range` into a list.
  axis = list(range(1, len(grad.get_shape())))
  avoid_zero_div = 1e-12
  if norm == np.inf:
    # Take sign of gradient
    optimal_perturbation = tf.sign(grad)
    # The following line should not change the numerical results. It applies only because
    # `optimal_perturbation` is the output of a `sign` op, which has zero derivative anyway.
    # It should not be applied for the other norms, where the perturbation has a non-zero derivative.
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
  elif norm == 1:
    abs_grad = tf.abs(grad)
    sign = tf.sign(grad)
    max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
    tied_for_max = tf.dtypes.cast(tf.equal(abs_grad, max_abs_grad), dtype=tf.float32)
    num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties
  elif norm == 2:
    square = tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
  scaled_perturbation = tf.multiply(eps, optimal_perturbation)
  return scaled_perturbation
