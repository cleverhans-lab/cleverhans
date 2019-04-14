"""Basic utilities for pytorch code"""

import warnings
from random import getrandbits

import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable


# https://gist.github.com/kingspp/3ec7d9958c13b94310c1a365759aa3f4
# Pyfunc Gradient Function
def _py_func_with_gradient(func, inp, Tout, stateful=True, name=None,
                           grad_func=None):
  """
  PyFunc defined as given by Tensorflow
  :param func: Custom Function
  :param inp: Function Inputs
  :param Tout: Ouput Type of out Custom Function
  :param stateful: Calculate Gradients when stateful is True
  :param name: Name of the PyFunction
  :param grad: Custom Gradient Function
  :return:
  """
  # Generate random name in order to avoid conflicts with inbuilt names
  rnd_name = 'PyFuncGrad-' + '%0x' % getrandbits(30 * 4)

  # Register Tensorflow Gradient
  tf.RegisterGradient(rnd_name)(grad_func)

  # Get current graph
  g = tf.get_default_graph()

  # Add gradient override map
  with g.gradient_override_map({"PyFunc": rnd_name,
                                "PyFuncStateless": rnd_name}):
    return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def convert_pytorch_model_to_tf(model, out_dims=None):
  """
  Convert a pytorch model into a tensorflow op that allows backprop
  :param model: A pytorch nn.Module object
  :param out_dims: The number of output dimensions (classes) for the model
  :return: A model function that maps an input (tf.Tensor) to the
  output of the model (tf.Tensor)
  """
  warnings.warn("convert_pytorch_model_to_tf will be deprecated in CleverHans"
                + " v4 as v4 will provide dedicated PyTorch support.")

  torch_state = {
      'logits': None,
      'x': None,
  }
  if not out_dims:
    out_dims = list(model.modules())[-1].out_features

  def _fprop_fn(x_np):
    """TODO: write this"""
    x_tensor = torch.Tensor(x_np)
    if torch.cuda.is_available():
      x_tensor = x_tensor.cuda()
    torch_state['x'] = Variable(x_tensor, requires_grad=True)
    torch_state['logits'] = model(torch_state['x'])
    return torch_state['logits'].data.cpu().numpy()

  def _bprop_fn(x_np, grads_in_np):
    """TODO: write this"""
    _fprop_fn(x_np)

    grads_in_tensor = torch.Tensor(grads_in_np)
    if torch.cuda.is_available():
      grads_in_tensor = grads_in_tensor.cuda()

    # Run our backprop through our logits to our xs
    loss = torch.sum(torch_state['logits'] * grads_in_tensor)
    loss.backward()
    return torch_state['x'].grad.cpu().data.numpy()

  def _tf_gradient_fn(op, grads_in):
    """TODO: write this"""
    return tf.py_func(_bprop_fn, [op.inputs[0], grads_in],
                      Tout=[tf.float32])

  def tf_model_fn(x_op):
    """TODO: write this"""
    out = _py_func_with_gradient(_fprop_fn, [x_op], Tout=[tf.float32],
                                 stateful=True,
                                 grad_func=_tf_gradient_fn)[0]
    out.set_shape([None, out_dims])
    return out

  return tf_model_fn


def clip_eta(eta, ord, eps):
  """
  PyTorch implementation of the clip_eta in utils_tf.

  :param eta: Tensor
  :param ord: np.inf, 1, or 2
  :param eps: float
  """
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')

  avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
  reduc_ind = list(range(1, len(eta.size())))
  if ord == np.inf:
    eta = torch.clamp(eta, -eps, eps)
  else:
    if ord == 1:
      # TODO
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


def optimize_linear(grad, eps, ord=np.inf):
  """
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)

  :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
  :param eps: float. Scalar specifying size of constraint region
  :param ord: np.inf, 1, or 2. Order of norm constraint.
  :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
  """

  red_ind = list(range(1, len(grad.size())))
  avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = torch.sign(grad)
  elif ord == 1:
    abs_grad = torch.abs(grad)
    sign = torch.sign(grad)
    red_ind = list(range(1, len(grad.size())))
    abs_grad = torch.abs(grad)
    ori_shape = [1]*len(grad.size())
    ori_shape[0] = grad.size(0)

    max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
    max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
    num_ties = torch.sum(max_mask, red_ind, keepdim=True)
    optimal_perturbation = sign * max_mask / num_ties
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
    assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
  elif ord == 2:
    square = torch.max(
        avoid_zero_div,
        torch.sum(grad ** 2, red_ind, keepdim=True)
        )
    optimal_perturbation = grad / torch.sqrt(square)
    # TODO integrate below to a test file
    # check that the optimal perturbations have been correctly computed
    opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
    one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + \
            (square > avoid_zero_div).to(torch.float)
    assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = eps * optimal_perturbation
  return scaled_perturbation
