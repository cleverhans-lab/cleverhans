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
  warnings.warn("convert_pytorch_model_to_tf is deprecated, switch to"
                + " dedicated PyTorch support provided by CleverHans v4.")

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
