"""The Projected Gradient Descent attack."""
import torch
import warnings

import numpy as np

from cleverhans.future.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.utils_pytorch import clip_eta


def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, ord,
                               clip_min=None, clip_max=None, y=None, targeted=False,
                               rand_init=None, rand_minmax=0.3, sanity_checks=True):
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param eps_iter: step size for each attack iteration
  :param nb_iter: Number of attack iterations.
  :param ord: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
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
  assert eps_iter <= eps, (eps_iter, eps)
  if y is not None and y_target is not None:
    raise ValueError("Must not set both y and y_target")
  if ord == 1:
    raise NotImplementedError("It's not clear that FGM is a good inner loop"
                              " step for PGD when ord=1, because ord=1 FGM "
                              " changes only one pixel at a time. We need "
                              " to rigorously test a strong ord=1 PGD "
                              "before enabling this feature.")
  if ord not in [np.inf, 2]:
    raise ValueError("Norm order must be either np.inf or 2.")

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # Initialize loop variables
  if rand_init:
    rand_minmax = eps
    eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
  else:
    eta = torch.zeros_like(x)

  # Clip eta
  eta = clip_eta(eta, ord, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  i = 0
  while i < nb_iter:
    adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, ord,
                                 clip_min=clip_min, clip_max=clip_max, y=y, targeted=targeted)

    # Clipping perturbation eta to ord norm ball
    eta = adv_x - x
    eta = clip_eta(eta, ord, eps)
    adv_x = x + eta

    # Redo the clipping.
    # FGM already did it, but subtracting and re-adding eta can add some
    # small numerical error.
    if clip_min is not None or clip_max is not None:
      adv_x = torch.clamp(adv_x, clip_min, clip_max)
    i += 1

  asserts.append(eps_iter <= eps)
  if ord == np.inf and clip_min is not None:
    # TODO necessary to cast clip_min and clip_max to x.dtype?
    asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x
