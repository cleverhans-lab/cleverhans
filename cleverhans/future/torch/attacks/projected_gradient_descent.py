"""The Projected Gradient Descent attack."""
import torch
import warnings

import numpy as np

from cleverhans.future.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.utils_pytorch import clip_eta

def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter,
                 y=None, ord=np.inf, clip_min=None, clip_max=None, y_target=None,
                 rand_init=None, rand_minmax=0.3, sanity_checks=True):
  """
  :param x: Tensor, shape (N, d_1, ...).
  :param kwargs: See `parse_params`
  """
  asserts = []

  if rand_init:
    rand_minmax = eps

  assert eps_iter <= eps, (eps_iter, eps)

  if y is not None and y_target is not None:
    raise ValueError("Must not set both y and y_target")
  # Check if order of the norm is acceptable given current implementation
  if ord not in [np.inf, 1, 2]:
    raise ValueError("Norm order must be either np.inf, 1, or 2.")
  sanity_checks = sanity_checks

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
  # Initialize loop variables
  if rand_init:
    eta = torch.zeros_like(x).uniform_(
        -rand_minmax,
        rand_minmax
    )
  else:
    eta = torch.zeros_like(x)

  # Clip eta
  # TODO clip_eta or optimize_linear?
  eta = clip_eta(eta, ord, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if y_target is not None:
    y = y_target
    targeted = True
  elif y is not None:
    y = y
    targeted = False
  else:
    _, y = torch.max(model_fn(x), 1)
    targeted = False

  y_kwarg = 'y_target' if targeted else 'y'
  fgm_params = {
      y_kwarg: y,
      'clip_min': clip_min,
      'clip_max': clip_max
  }
  # TODO ignoring this for testing purposes
  """
  if ord == 1:
      raise NotImplementedError("It's not clear that FGM is a good inner loop"
                              " step for PGD when ord=1, because ord=1 FGM "
                              " changes only one pixel at a time. We need "
                              " to rigorously test a strong ord=1 PGD "
                              "before enabling this feature.")
  """
  i = 0
  while i < nb_iter:
    adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, ord, **fgm_params)

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
    # TODO necessary to cast to x.dtype?
    asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
    assert np.all(asserts)
  return adv_x

