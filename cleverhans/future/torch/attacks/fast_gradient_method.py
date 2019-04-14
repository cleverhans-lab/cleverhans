"""
The Fast Gradient Method attack. (PyTorch implementation)
"""
# currently tested on pytorch v1.0, numpy v1.16, python 3.6

import warnings

import numpy as np
import torch

from cleverhans.utils_pytorch import get_or_guess_labels, optimize_linear

class FastGradientMethod:
  """
  This attack was originally implemented by Goodfellow et al. (2014) with the
  infinity norm (and is known as the "Fast Gradient Sign Method"). This
  implementation extends the attack to other norms, and is therefore called
  the Fast Gradient Method.
  Paper link: https://arxiv.org/abs/1412.6572

  :param model: PyTorch model. Do not add a softmax gate to the output
  :param dtype: (optional) dtype of the data
  """

  def __init__(self, model, dtype=torch.float32):
    """
    Create a FastGradientMethod instance.
    """
    self.model = model
    self.dtype = dtype

  def generate(self, x, **kwargs):
    """
    Generate adversarial examples for x using FGM.

    :param x: Tensor
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)
    labels = get_or_guess_labels(model=self.model, x=x, **kwargs)

    # set the model to evaluation mode to have DropOut and BatchNorm layers
    # behave properly
    self.model.eval()

    return fgm(
        x=x,
        model=self.model,
        y=labels,
        eps=self.eps,
        ord=self.ord,
        clip_min=self.clip_min,
        clip_max=self.clip_max,
        targeted=(self.y_target is not None),
        sanity_checks=self.sanity_checks
        )

  def parse_params(self,
                   eps=0.3,
                   ord=np.inf,
                   y=None,
                   y_target=None,
                   clip_min=None,
                   clip_max=None,
                   sanity_checks=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional) float. Attack step size (input variation)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param y: (optional) Tensor, shape (N), where N is the batch size.
              A tensor with true labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
    :param y_target: (optional) Tensor, shape (N).
                    A tensor with the labels to target. Leave
                     y_target=None if y is also set.
    :param clip_min: (optional) float. Minimum input component value
    :param clip_max: (optional) float. Maximum input component value
    :param sanity_checks: bool. If True, include asserts
      (Turn them off to use less runtime / memory or for unit tests that
      intentionally pass strange input)
    """
    # Save attack-specific parameters

    self.eps = eps
    self.ord = ord
    self.y = y
    self.y_target = y_target
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, int(1), int(2)]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


def fgm(x,
        model,
        y=None,
        eps=0.3,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False,
        sanity_checks=True):
  """
  PyTorch implementation of the Fast Gradient Method.
  :param x: Tensor, shape (N, d_1, ...).
  :param logits: Tensor, shape (N, C). Raw output of model(x), not softmax
            probability.
  :param y: (optional) Tensor, shape (N). True labels. If targeted
            is true, then provide the target label. Otherwise, only provide
            this parameter if you'd like to use true labels when crafting
            adversarial samples. Otherwise, model predictions are used as
            labels to avoid the "label leaking" effect (explained in this
            paper: https://arxiv.org/abs/1611.01236). Default is None.
  :param eps: (optional) float. The epsilon (input variation parameter)
  :param ord: (optional) Order of the norm (mimics NumPy).
              Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial
            example components
  :param clip_max: (optional) float. Maximum float value for adversarial
            example components
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more
                   like y.
  :return: a tensor for the adversarial example
  """
  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(torch.all(torch.ge(
        x,
        torch.tensor(clip_min, device=x.device, dtype=x.dtype)
    )))

  if clip_max is not None:
    asserts.append(torch.all(torch.le(
        x,
        torch.tensor(clip_max, device=x.device, dtype=x.dtype)
    )))

  # x needs to be a leaf variable, of floating point type and have requires_grad being True for
  # its grad to be computed and stored properly in a backward call
  x = x.clone().detach().to(torch.float).requires_grad_(True)
  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model(x), 1)

  # Compute loss
  loss_fn = torch.nn.CrossEntropyLoss()
  loss = loss_fn(model(x), y)
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  loss.backward()
  optimal_perturbation = optimize_linear(x.grad, eps, ord)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + optimal_perturbation

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x
