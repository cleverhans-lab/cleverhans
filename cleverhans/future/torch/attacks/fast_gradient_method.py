"""
The FastGradientMethod attack. (PyTorch implementation)
"""
# currently tested on pytorch v1.0, numpy v1.16, python 3.6

import warnings

import numpy as np
import torch

from cleverhans.future.torch.attacks.attack import Attack

class FastGradientMethod(Attack):
  """
  This attack was originally implemented by Goodfellow et al. (2014) with the
  infinity norm (and is known as the "Fast Gradient Sign Method"). This
  implementation extends the attack to other norms, and is therefore called
  the Fast Gradient Method.
  Paper link: https://arxiv.org/abs/1412.6572

  :param model: PyTorch model. Needs to have model.forward implemented with
  output being raw probability (logits)
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, dtype=torch.float32, **kwargs):
    """
    Create a FastGradientMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(FastGradientMethod, self).__init__(model, dtype, **kwargs)
    self.feedable_kwargs = ('eps', 'y', 'y_target', 'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'sanity_checks']

  def generate(self, x, **kwargs):
    """
    Returns the graph for Fast Gradient Method adversarial examples.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    labels = self.get_or_guess_labels(x, kwargs)

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
        sanity_checks=self.sanity_checks)

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

    :param eps: (optional float) attack step size (input variation)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param y: (optional) A tensor with the true labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool, if True, include asserts
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
  TensorFlow implementation of the Fast Gradient Method.
  :param x: the input placeholder
  :param logits: output of model.get_logits
  :param y: (optional) A placeholder for the true labels. If targeted
            is true, then provide the target label. Otherwise, only provide
            this parameter if you'd like to use true labels when crafting
            adversarial samples. Otherwise, model predictions are used as
            labels to avoid the "label leaking" effect (explained in this
            paper: https://arxiv.org/abs/1611.01236). Default is None.
            Labels should be one-hot-encoded.
  :param eps: the epsilon (input variation parameter)
  :param ord: (optional) Order of the norm (mimics NumPy).
              Possible values: np.inf, 1 or 2.
  :param clip_min: Minimum float value for adversarial example components
  :param clip_max: Maximum float value for adversarial example components
  :param targeted: Is the attack targeted or untargeted? Untargeted, the
                   default, will try to make the label incorrect. Targeted
                   will instead try to move in the direction of being more
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

  # TODO doc: make sure the caller has not passed probs by accident

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


def optimize_linear(grad, eps, ord=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)

  :param grad: tf tensor containing a batch of gradients
  :param eps: float scalar specifying size of constraint region
  :param ord: int specifying order of norm
  :returns:
    tf tensor containing optimal perturbation
  """

  # In Python 2, the `list` call in the following line is redundant / harmless.
  # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
  red_ind = list(range(1, len(grad.size())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = torch.sign(grad)
  elif ord == 1:
    abs_grad = torch.abs(grad)
    sign = torch.sign(grad)
    red_ind = list(range(1, len(grad.size())))
    abs_grad = torch.abs(grad)

    max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1, keepdim=True)
    max_mask = abs_grad.eq(max_abs_grad).to(torch.float)
    num_ties = torch.sum(max_mask, red_ind)
    optimal_perturbation = sign * max_mask / num_ties
  elif ord == 2:
    square = torch.max(
        avoid_zero_div,
        torch.sum(grad ** 2, red_ind, keepdim=True)
        )
    optimal_perturbation = grad / torch.sqrt(square)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = eps * optimal_perturbation
  return scaled_perturbation
