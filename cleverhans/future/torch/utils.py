"""Utils for PyTorch"""

import numpy as np

import torch


def clip_eta(eta, norm, eps):
  """
  PyTorch implementation of the clip_eta in utils_tf.

  :param eta: Tensor
  :param norm: np.inf, 1, or 2
  :param eps: float
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError('norm must be np.inf, 1, or 2.')

  avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
  reduc_ind = list(range(1, len(eta.size())))
  if norm == np.inf:
    eta = torch.clamp(eta, -eps, eps)
  else:
    if norm == 1:
      raise NotImplementedError("L1 clip is not implemented.")
      norm = torch.max(
          avoid_zero_div,
          torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
      )
    elif norm == 2:
      norm = torch.sqrt(torch.max(
          avoid_zero_div,
          torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
      ))
    factor = torch.min(
        torch.tensor(1., dtype=eta.dtype, device=eta.device),
        eps / norm
        )
    eta *= factor
  return eta

def get_or_guess_labels(model, x, **kwargs):
  """
  Get the label to use in generating an adversarial example for x.
  The kwargs are fed directly from the kwargs of the attack.
  If 'y' is in kwargs, then assume it's an untargeted attack and
  use that as the label.
  If 'y_target' is in kwargs and is not none, then assume it's a
  targeted attack and use that as the label.
  Otherwise, use the model's prediction as the label and perform an
  untargeted attack.

  :param model: PyTorch model. Do not add a softmax gate to the output.
  :param x: Tensor, shape (N, d_1, ...).
  :param y: (optional) Tensor, shape (N).
  :param y_target: (optional) Tensor, shape (N).
  """
  if 'y' in kwargs and 'y_target' in kwargs:
    raise ValueError("Can not set both 'y' and 'y_target'.")
  if 'y' in kwargs:
    labels = kwargs['y']
  elif 'y_target' in kwargs and kwargs['y_target'] is not None:
    labels = kwargs['y_target']
  else:
    _, labels = torch.max(model(x), 1)
  return labels


def optimize_linear(grad, eps, norm=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

  :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
  :param eps: float. Scalar specifying size of constraint region
  :param norm: np.inf, 1, or 2. Order of norm constraint.
  :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
  """

  red_ind = list(range(1, len(grad.size())))
  avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
  if norm == np.inf:
    # Take sign of gradient
    optimal_perturbation = torch.sign(grad)
  elif norm == 1:
    abs_grad = torch.abs(grad)
    sign = torch.sign(grad)
    red_ind = list(range(1, len(grad.size())))
    abs_grad = torch.abs(grad)
    ori_shape = [1]*len(grad.size())
    ori_shape[0] = grad.size(0)

    max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
    max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
    num_ties = max_mask
    for red_scalar in red_ind:
      num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
    optimal_perturbation = sign * max_mask / num_ties
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
    assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
  elif norm == 2:
    square = torch.max(
        avoid_zero_div,
        torch.sum(grad ** 2, red_ind, keepdim=True)
        )
    optimal_perturbation = grad / torch.sqrt(square)
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
    one_mask = (
        (square <= avoid_zero_div).to(torch.float) * opt_pert_norm +
        (square > avoid_zero_div).to(torch.float))
    assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = eps * optimal_perturbation
  return scaled_perturbation
