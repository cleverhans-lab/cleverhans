"""
MultiGPU model similar to the one used in model tutorials. The model keeps
one copy of the weights on each device and handles syncing the parameters
across devices.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from cleverhans.model import Model


def clone_variable(name, x, trainable=False):
  return tf.get_variable(name, shape=x.shape, dtype=x.dtype,
                         trainable=trainable)


def unify_device_name(dname):
  """Converts TensorFlow device names in the format /Device:GPU0 to /gpu:0.
  """
  if dname is None:
    return None
  return dname.lower().replace('device:', '')


class MLP(Model):
  """
  An example of a bare bones multilayer perceptron (MLP) class.
  """

  def __init__(self, nb_classes, layers, input_shape):
    super(MLP, self).__init__('MLP', nb_classes=nb_classes, hparams={})

    self.layer_names = []
    self.layers = layers
    self.input_shape = input_shape
    if isinstance(layers[-1], Softmax):
      layers[-1].name = 'probs'
      layers[-2].name = 'logits'
    else:
      layers[-1].name = 'logits'
    for i, layer in enumerate(self.layers):
      if hasattr(layer, 'name'):
        name = layer.name
      else:
        name = layer.__class__.__name__ + str(i)
        layer.name = name
      self.layer_names.append(name)

      layer.set_input_shape(input_shape)
      input_shape = layer.get_output_shape()

  def fprop(self, x, set_ref=False):
    states = []
    for layer in self.layers:
      if set_ref:
        layer.ref = x
      x = layer.fprop(x)
      assert x is not None
      states.append(x)
    states = dict(zip(self.layer_names, states))
    return states

  def get_params(self):
    out = []
    for layer in self.layers:
      for param in layer.get_params():
        if param not in out:
          out.append(param)
    return out


class Layer(object):

  def get_output_shape(self):
    return self.output_shape


class Linear(Layer):

  def __init__(self, num_hid):
    self.num_hid = num_hid

  def set_input_shape(self, input_shape):
    batch_size, dim = input_shape
    self.input_shape = [batch_size, dim]
    self.output_shape = [batch_size, self.num_hid]
    init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
    init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                               keep_dims=True))
    self.W = tf.Variable(init)
    self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

  def fprop(self, x):
    return tf.matmul(x, self.W) + self.b

  def get_params(self):
    return [self.W, self.b]


class Conv2D(Layer):

  def __init__(self, output_channels, kernel_shape, strides, padding):
    self.__dict__.update(locals())
    del self.self

  def set_input_shape(self, input_shape):
    batch_size, _, __, input_channels = input_shape
    kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                               self.output_channels)
    assert len(kernel_shape) == 4
    assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
    init = tf.random_normal(kernel_shape, dtype=tf.float32)
    init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                               axis=(0, 1, 2)))
    self.kernels = tf.Variable(init)
    self.b = tf.Variable(
        np.zeros((self.output_channels,)).astype('float32'))
    input_shape = list(input_shape)
    input_shape[0] = 1
    dummy_batch = tf.zeros(input_shape)
    dummy_output = self.fprop(dummy_batch)
    output_shape = [int(e) for e in dummy_output.get_shape()]
    output_shape[0] = batch_size
    self.output_shape = tuple(output_shape)

  def fprop(self, x):
    return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                        self.padding) + self.b

  def get_params(self):
    return [self.kernels, self.b]


class ReLU(Layer):

  def __init__(self):
    pass

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def fprop(self, x):
    return tf.nn.relu(x)

  def get_params(self):
    return []


class Softmax(Layer):

  def __init__(self):
    pass

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def fprop(self, x):
    return tf.nn.softmax(x)

  def get_params(self):
    return []


class Flatten(Layer):

  def __init__(self):
    pass

  def set_input_shape(self, shape):
    self.input_shape = shape
    output_width = 1
    for factor in shape[1:]:
      output_width *= factor
    self.output_width = output_width
    self.output_shape = [shape[0], output_width]

  def fprop(self, x):
    return tf.reshape(x, [-1, self.output_width])

  def get_params(self):
    return []


class MLPnGPU(MLP):
  """
  A multi layer perceptron that can be copied over multiple GPUs. Only one
  copy of the weights is created on each device.
  """

  def __init__(self, nb_classes, layers, input_shape):
    super(MLPnGPU, self).__init__(nb_classes, layers, input_shape)
    self.scope = 'MLPnGPU'

  def fprop(self, x):
    with tf.variable_scope(self.scope):
      states = super(MLPnGPU, self).fprop(x)
    return states

  def set_device(self, device_name):
    """
    Set the device before the next fprop to create a new graph on the
    specified device.
    """
    device_name = unify_device_name(device_name)
    self.device_name = device_name
    for layer in self.layers:
      layer.device_name = device_name

  def create_sync_ops(self, host_device):
    """
    Return a list of assignment operations that syncs the parameters
    of all model copies with the one on host_device.
    :param host_device: (required str) the name of the device with latest
                        parameters
    """
    host_device = unify_device_name(host_device)
    sync_ops = []
    for layer in self.layers:
      if isinstance(layer, LayernGPU):
        sync_ops += layer.create_sync_ops(host_device)
    return sync_ops

  def set_training(self, training=False):
    for layer in self.layers:
      if isinstance(layer, LayernGPU):
        layer.set_training(training)


class LayernGPU(Layer):
  """
  A layer that has separate copies of model parameters on each GPU.
  """

  def __init__(self):
    """
    :param input_shape: a tuple or list as the input shape to layer
    """
    self.input_shape = None
    self.params_device = {}
    self.params_names = None
    self.device_name = '/gpu:0'
    self.training = True

  def set_training(self, training=False):
    self.training = training

  def get_variable(self, name, initializer):
    """
    Create and initialize a variable using a numpy array and set trainable.
    :param name: (required str) name of the variable
    :param initializer: a numpy array or a tensor
    """
    v = tf.get_variable(name, shape=initializer.shape,
                        initializer=(lambda shape, dtype, partition_info:
                                     initializer),
                        trainable=self.training)
    return v

  def set_input_shape_ngpu(self, new_input_shape):
    """
    Create and initialize layer parameters on the device previously set
    in self.device_name.

    :param new_input_shape: a list or tuple for the shape of the input.
    """
    assert self.device_name, "Device name has not been set."

    device_name = self.device_name
    if self.input_shape is None:
      # First time setting the input shape
      self.input_shape = [None] + [int(d) for d in list(new_input_shape)]

    if device_name in self.params_device:
      # There is a copy of weights on this device
      self.__dict__.update(self.params_device[device_name])
      return

    # Stop recursion
    self.params_device[device_name] = {}

    # Initialize weights on this device
    with tf.device(device_name):
      self.set_input_shape(self.input_shape)
      keys_after = self.__dict__.keys()
      if self.params_names is None:
        # Prevent overriding training
        self.params_names = [k for k in keys_after if isinstance(
            self.__dict__[k], tf.Variable)]
      params = {k: self.__dict__[k] for k in self.params_names}
      self.params_device[device_name] = params

  def create_sync_ops(self, host_device):
    """Create an assignment operation for each weight on all devices. The
    weight is assigned the value of the copy on the `host_device'.
    """
    sync_ops = []
    host_params = self.params_device[host_device]
    for device, params in (self.params_device).iteritems():
      if device == host_device:
        continue
      for k in self.params_names:
        if isinstance(params[k], tf.Variable):
          sync_ops += [tf.assign(params[k], host_params[k])]
    return sync_ops

  def fprop(self, x):
    if self.name is None:
      self.set_input_shape_ngpu(x.shape[1:])
      return self.fprop_noscope(x)
    else:
      with tf.variable_scope(self.name):
        self.set_input_shape_ngpu(x.shape[1:])
        return self.fprop_noscope(x)


class LinearnGPU(LayernGPU):

  def __init__(self, num_hid, w_name='W'):
    super(LinearnGPU, self).__init__()
    self.num_hid = num_hid
    self.w_name = w_name

  def set_input_shape(self, input_shape):
    batch_size, dim = input_shape
    self.input_shape = [batch_size, dim]
    self.output_shape = [batch_size, self.num_hid]
    shape = [dim, self.num_hid]
    with tf.variable_scope(self.name):
      init = tf.truncated_normal(shape, stddev=0.1)
      self.W = self.get_variable(self.w_name, init)
      self.b = self.get_variable('b', .1 + np.zeros(
          (self.num_hid,)).astype('float32'))

  def fprop_noscope(self, x):
    return tf.matmul(x, self.W) + self.b


class Conv2DnGPU(LayernGPU):

  def __init__(self, output_channels, kernel_shape, strides, padding,
               w_name='kernels'):
    super(Conv2DnGPU, self).__init__()
    self.__dict__.update(locals())
    del self.self
    self.w_name = w_name

  def set_input_shape(self, input_shape):
    assert len(input_shape) == 4
    input_channels = input_shape[3]
    kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                               self.output_channels)
    assert len(kernel_shape) == 4
    assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
    with tf.variable_scope(self.name):
      init = tf.truncated_normal(kernel_shape, stddev=0.1)
      self.kernels = self.get_variable(self.w_name, init)
      self.b = self.get_variable(
          'b', .1 + np.zeros((self.output_channels,)).astype('float32'))
    input_shape = list(input_shape)
    self.input_shape = input_shape
    input_shape[0] = 1
    dummy_batch = tf.zeros(input_shape)
    dummy_output = self.fprop(dummy_batch)
    output_shape = [int(e) for e in dummy_output.get_shape()]
    output_shape[0] = 1
    self.output_shape = tuple(output_shape)

  def fprop_noscope(self, x):
    return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) +
                        (1,), self.padding) + self.b


class MaxPool(LayernGPU):
  def __init__(self, ksize, strides, padding):
    super(MaxPool, self).__init__()
    self.__dict__.update(locals())
    del self.self

  def set_input_shape(self, input_shape):
    input_shape = list(input_shape)
    input_shape[0] = 1
    dummy_batch = tf.zeros(input_shape)
    dummy_output = self.fprop(dummy_batch)
    output_shape = [int(e) for e in dummy_output.get_shape()]
    output_shape[0] = 1
    self.output_shape = tuple(output_shape)

  def fprop_noscope(self, x):
    return tf.nn.max_pool(x,
                          ksize=(1,) + tuple(self.ksize) + (1,),
                          strides=(1,) + tuple(self.strides) + (1,),
                          padding=self.padding)


class LayerNorm(LayernGPU):

  def set_input_shape(self, input_shape):
    self.input_shape = list(input_shape)
    params_shape = [input_shape[-1]]
    self.params_shape = params_shape

    self.beta = tf.get_variable(
        'beta', params_shape, tf.float32,
        initializer=tf.constant_initializer(0.0, tf.float32),
        trainable=self.training)
    self.gamma = tf.get_variable(
        'gamma', params_shape, tf.float32,
        initializer=tf.constant_initializer(1.0, tf.float32),
        trainable=self.training)

  def fprop_noscope(self, x):
    mean = tf.reduce_mean(x, (1, 2), keep_dims=True)
    x = x - mean
    std = tf.sqrt(1e-7 +
                  tf.reduce_mean(tf.square(x), (1, 2), keep_dims=True))
    x = x / std
    return x * self.gamma + self.beta
