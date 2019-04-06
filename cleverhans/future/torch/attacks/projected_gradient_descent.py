"""
The ProjectedGradientDescent attack (PyTorch implementation).
"""
# currently tested on pytorch v1.0, numpy v1.16, python 3.6

import warnings

import numpy as np
import torch

from cleverhans.attacks.fast_gradient_method import FastGradientMethod

class ProjectedGradientDescent:
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf

  :param model: PyTorch model. Needs to have model.forward implemented with
  output being raw probability (logits)
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param default_rand_init: whether to use random initialization by default
  :param kwargs: passed through to super constructor
  """

  FGM_CLASS = FastGradientMethod

  def __init__(self, model, dtype=torch.float32,
               default_rand_init=True, **kwargs):
    """
    Create a ProjectedGradientDescent instance.
    """

    super(ProjectedGradientDescent, self).__init__(model,
                                                   dtype=dtype, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                            'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'sanity_checks']
    self.default_rand_init = default_rand_init

  def generate(self, x, **kwargs):
    """
    :param x: Input data. Should be a PyTorch Tensor.
    :param kwargs: See `parse_params`
    """
    # TODO support numpy tensors. Wrap x as a pytorch tensor before excuting?

    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    asserts = []

    # If a data range was specified, check that the input was in that range
    if self.clip_min is not None:
      asserts.append(torch.all(torch.ge(
          x,
          torch.tensor(self.clip_min, device=x.device, dtype=x.dtype)
      )))

    if self.clip_max is not None:
      asserts.append(torch.all(torch.le(
          x,
          torch.tensor(self.clip_max, device=x.device, dtype=x.dtype)
      )))
    # Initialize loop variables
    if self.rand_init:
      eta = torch.zeros_like(x).uniform_(
          -self.rand_minmax,
          self.rand_minmax
      )
    else:
      eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, self.ord, self.eps)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      adv_x = torch.clamp(adv_x, self.clip_min, self.clip_max)

    if self.y_target is not None:
      y = self.y_target
      targeted = True
    elif self.y is not None:
      y = self.y
      targeted = False
    else:
      _, y = torch.max(self.model(x), 1)
      targeted = False

    y_kwarg = 'y_target' if targeted else 'y'
    fgm_params = {
        'eps': self.eps_iter,
        y_kwarg: y,
        'ord': self.ord,
        'clip_min': self.clip_min,
        'clip_max': self.clip_max
    }
    if self.ord == 1:
      raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                " step for PGD when ord=1, because ord=1 FGM "
                                " changes only one pixel at a time. We need "
                                " to rigorously test a strong ord=1 PGD "
                                "before enabling this feature.")

    # TODO pytorch implementation of FGM_CLASS
    FGM = self.FGM_CLASS(
        self.model,
        dtype=self.dtype
        )
    i = 0
    while i < self.nb_iter:
      adv_x = FGM.generate(adv_x, **fgm_params)

      # Clipping perturbation eta to self.ord norm ball
      eta = adv_x - x
      eta = clip_eta(eta, self.ord, self.eps)
      adv_x = x + eta

      # Redo the clipping.
      # FGM already did it, but subtracting and re-adding eta can add some
      # small numerical error.
      if self.clip_min is not None or self.clip_max is not None:
        adv_x = torch.clamp(adv_x, self.clip_min, self.clip_max)

    asserts.append(self.eps_iter <= self.eps)
    if self.ord == np.inf and self.clip_min is not None:
      # The 1e-6 is needed to compensate for numerical error.
      # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5,
      # clip_max=.7
      # TODO case to x.dtype?
      asserts.append(self.eps <= (1e-6 + self.clip_max - self.clip_min))

    if self.sanity_checks:
      assert np.all(asserts)
    return adv_x

  def parse_params(self,
                   eps=0.3,
                   eps_iter=0.05,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   rand_init=None,
                   rand_minmax=0.3,
                   sanity_checks=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the true labels.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool Insert tf asserts checking values
        (Some tests need to run with no sanity checks because the
         tests intentionally configure the attack strangely)
    """

    # Save attack-specific parameters
    self.eps = eps
    if rand_init is None:
      rand_init = self.default_rand_init
    self.rand_init = rand_init
    if self.rand_init:
      self.rand_minmax = eps
    else:
      self.rand_minmax = 0.
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    if isinstance(eps, float) and isinstance(eps_iter, float):
      # If these are both known at compile time, we can check before anything
      # is run. If they are tf, we can't check them yet.
      assert eps_iter <= eps, (eps_iter, eps)

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")
    self.sanity_checks = sanity_checks

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True

def clip_eta(eta, ord, eps):
  """
  PyTorch implementation of the clip_eta in utils_tf.
  :param eta: Tensor
  """
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')

  avoid_zero_div = torch.tensor(1e-12)
  reduc_ind = list(range(1, len(eta.size())))
  if ord == np.inf:
    eta = torch.clamp(eta, -eps, eps)
  else:
    if ord == 1:
      # raise NotImplementedError("L1 clip is not implemented.")
      norm = torch.max(
          avoid_zero_div,
          torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
      )
    elif ord == 2:
      norm = torch.sqrt(torch.max(
          avoid_zero_div,
          torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
      ))
    factor = torch.min(torch.tensor(1.), eps / norm)
    eta *= factor
  return eta
