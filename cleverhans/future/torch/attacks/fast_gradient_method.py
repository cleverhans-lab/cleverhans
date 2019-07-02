"""The Fast Gradient Method attack."""
import numpy as np
import torch

from cleverhans.future.torch.utils import optimize_linear


def fast_gradient_method(model_fn, x, eps, norm,
                         clip_min=None, clip_max=None, y=None, targeted=False, sanity_checks=False):
  """
  PyTorch implementation of the Fast Gradient Method.
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
    raise ValueError("Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm))
  if eps < 0:
    raise ValueError("eps must be greater than or equal to 0, got {} instead".format(eps))
  if eps == 0:
    return x
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
              clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # x needs to be a leaf variable, of floating point type and have requires_grad being True for
  # its grad to be computed and stored properly in a backward call
  x = x.clone().detach().to(torch.float).requires_grad_(True)
  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  # Compute loss
  loss_fn = torch.nn.CrossEntropyLoss()
  loss = loss_fn(model_fn(x), y)
  # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  loss.backward()
  optimal_perturbation = optimize_linear(x.grad, eps, norm)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    if clip_min is None or clip_max is None:
      raise ValueError(
          "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x
