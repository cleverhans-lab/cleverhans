"""Models that support pickling.

NOTE: This module is much more free to change than many other modules
in CleverHans. CleverHans is very conservative about changes to any
code that affects the output of benchmark tests (attacks, evaluation
methods, etc.). This module provides *models* not *benchmarks* and
thus is free to change rapidly to provide better speed, accuracy,
etc.
"""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from cleverhans.compat import reduce_mean, reduce_prod
from cleverhans.model import Model
from cleverhans.serial import PicklableVariable as PV
from cleverhans.utils import ordered_union


class PicklableModel(Model):
  """
  A Model that supports pickling.

  Subclasses of this model must use only PicklableVariable and must refer
  to their variables only by referencing the Python objects, not using
  TensorFlow names (so no variable scopes). Pickle cannot find variables
  referenced only by name and thus may fail to save them. Pickle may not
  be able to get the original name back when restoring the variable so the
  names should not be relied on.
  """

  def __init__(self):
    super(PicklableModel, self).__init__()
    del self.scope  # Must not use Variable scopes / names for anything

  def get_params(self):
    raise NotImplementedError(str(type(self)) + " does not implement"
                              " get_params.")


class MLP(PicklableModel):
  """
  A picklable multilayer perceptron
  """

  def __hash__(self):
    return hash(id(self))

  def __init__(self, layers, input_shape):
    super(MLP, self).__init__()

    if not isinstance(layers, list):
      raise ValueError("`layers` must be a list.")

    self.layer_names = []
    self.layers = layers
    self.input_shape = input_shape

    if isinstance(layers[-1], Softmax):
      if not hasattr(layers[-1], 'name'):
        layers[-1].name = 'probs'
      if not hasattr(layers[-2], 'name'):
        layers[-2].name = 'logits'
    else:
      if not hasattr(layers[-1], 'name'):
        layers[-1].name = 'logits'
    for i, layer in enumerate(self.layers):
      if layer.parent is None:
        if i == 0:
          layer.parent = "input"
        else:
          layer.parent = layers[i - 1].name
      if hasattr(layer, 'name'):
        name = layer.name
      else:
        name = layer.__class__.__name__ + str(i)
        layer.name = name
      self.layer_names.append(name)

      layer.set_input_shape(input_shape)
      input_shape = layer.get_output_shape()

  def get_params(self):
    out = []
    for layer in self.layers:
      out = ordered_union(out, layer.get_params())
    return out

  def fprop(self, x=None, given=None, **kwargs):

    # Note: this currently isn't great.
    # A layer can have any parent it wants, but the parent
    # must come earlier in the list.
    # There's no way to have > 1 parent.
    # This means we can support branched structures that split,
    # e.g. for multiple output heads, but not structures
    # that converge.
    # We can feed a value in the middle using "given" but
    # only layers after the given one are run using the current
    # implementation, so the feed must happen before any branch
    # point.

    if x is None:
      if given is None:
        raise ValueError("One of `x` or `given` must be specified")
    else:
      assert given is None
      given = ('input', x)
    name, value = given
    out = {name: value}
    x = value

    if name == 'input':
      layers = self.layers
    else:
      for i, layer in enumerate(self.layers[:-1]):
        if layer.name == name:
          layers = self.layers[i+1:]
          break

    for layer in layers:
      x = out[layer.parent]
      try:
        x = layer.fprop(x, **kwargs)
      except TypeError as e:
        msg = "TypeError in fprop for %s of type %s: %s"
        msg = msg % (layer.name, str(type(layer)), str(e))
        raise TypeError(msg)
      assert x is not None
      out[layer.name] = x
    return out

  def make_input_placeholder(self):
    return tf.placeholder(tf.float32, tuple(self.input_shape))

  def make_label_placeholder(self):
    try:
      return self.layers[-1].make_label_placeholder()
    except NotImplementedError:
      return tf.placeholder(tf.float32,
                            self.layers[-1].get_output_shape())


class Layer(PicklableModel):
  def __init__(self, name=None, parent=None):
    super(Layer, self).__init__()
    if name is not None:
      self.name = name
    self.parent = parent

  def get_output_shape(self):
    return self.output_shape


class Linear(Layer):
  """
  Linear, fully connected layer.
  :param init_mode: string
      "norm" : the weight vector for each output is initialized to have
               the same norm, given by `init_scale`
     "uniform_unit_scaling" :  U(-sqrt(3/input_dim), sqrt(3/input_dim))
          from https://arxiv.org/abs/1412.6558
  """

  def __init__(self, num_hid, init_scale=1., init_b=0., use_bias=True,
               init_mode="norm",
               **kwargs):
    super(Linear, self).__init__(**kwargs)
    self.num_hid = num_hid
    self.init_scale = init_scale
    self.init_b = init_b
    self.use_bias = use_bias
    self.init_mode = init_mode

  def set_input_shape(self, input_shape):
    batch_size, dim = input_shape
    self.input_shape = [batch_size, dim]
    self.output_shape = [batch_size, self.num_hid]
    if self.init_mode == "norm":
      init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
      init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                 keep_dims=True))
      init = init * self.init_scale
    elif self.init_mode == "uniform_unit_scaling":
      scale = np.sqrt(3. / dim)
      init = tf.random_uniform([dim, self.num_hid], dtype=tf.float32,
                               minval=-scale, maxval=scale)
    else:
      raise ValueError(self.init_mode)
    self.W = PV(init)
    if self.use_bias:
      self.b = PV((np.zeros((self.num_hid,))
                   + self.init_b).astype('float32'))

  def fprop(self, x, **kwargs):
    out = tf.matmul(x, self.W.var)
    if self.use_bias:
      out = out + self.b.var
    return out

  def get_params(self):
    out = [self.W.var]
    if self.use_bias:
      out.append(self.b.var)
    return out


class Conv2D(Layer):
  """
  2-D Convolution. Uses NHWC format for input and output.
  :param output_channels: int
      The number of channels to output
  :param kernel_shape: tuple of two ints
      (kernel rows, kernel columns)
      Do not include input channels or output channels in kernel_shape.
  :param strides: tuple of two ints
      (row stride, column stride)
      Do not include channel or batch strides.
  :param use_bias: bool
      If True (default is False) adds a per-channel bias term to the output
  :param init_mode: string
      "norm" : each kernel is initialized to have the same norm,
               given by `init_scale`
     "inv_sqrt" :  Gaussian with standard devation given by sqrt(2/fan_out)
     "glorot_uniform" : U(+/- sqrt(6/(fan_in+fan_out))
  """

  def __init__(self, output_channels, kernel_shape, strides, padding,
               use_bias=False, init_scale=1.,
               init_mode="norm", **kwargs):
    self.__dict__.update(locals())
    del self.self
    super(Conv2D, self).__init__(**kwargs)

  def set_input_shape(self, input_shape):
    _batch_size, _rows, _cols, input_channels = input_shape
    assert len(self.kernel_shape) == 2
    kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                               self.output_channels)
    assert len(kernel_shape) == 4
    assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
    fan_in = self.kernel_shape[0] * \
        self.kernel_shape[1] * input_channels
    fan_out = self.kernel_shape[0] * \
        self.kernel_shape[1] * self.output_channels
    if self.init_mode == "norm":
      init = tf.random_normal(kernel_shape, dtype=tf.float32)
      squared_norms = tf.reduce_sum(tf.square(init), axis=(0, 1, 2))
      denom = tf.sqrt(1e-7 + squared_norms)
      init = self.init_scale * init / denom
    elif self.init_mode == "inv_sqrt":
      init = tf.random_normal(kernel_shape, dtype=tf.float32,
                              stddev=np.sqrt(2.0 / fan_out))
    elif self.init_mode == "glorot_uniform":
      scale = np.sqrt(6. / (fan_in + fan_out))
      init = tf.random_uniform(kernel_shape, dtype=tf.float32,
                               minval=-scale, maxval=scale)
    else:
      raise ValueError(self.init_mode)
    self.kernels = PV(init, name=self.name + "_kernels")
    if self.use_bias:
      self.b = PV(np.zeros((self.output_channels,)).astype('float32'))
    input_shape = list(input_shape)
    orig_batch_size = input_shape[0]
    input_shape[0] = 1
    dummy_batch = tf.zeros(input_shape)
    dummy_output = self.fprop(dummy_batch)
    output_shape = [int(e) for e in dummy_output.get_shape()]
    output_shape[0] = orig_batch_size
    self.output_shape = tuple(output_shape)

  def fprop(self, x, **kwargs):
    assert len(self.strides) == 2
    out = tf.nn.conv2d(x, self.kernels.var,
                       (1,) + tuple(self.strides) + (1,), self.padding)
    if self.use_bias:
      out = out + self.b.var
    return out

  def get_params(self):
    out = [self.kernels.var]
    if self.use_bias:
      out.append(self.b.var)
    return out


class ReLU(Layer):

  def __init__(self, leak=0., **kwargs):
    super(ReLU, self).__init__(**kwargs)
    self.leak = leak

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_output_shape(self):
    return self.output_shape

  def fprop(self, x, **kwargs):
    out = tf.nn.relu(x)
    if self.leak != 0.0:
      # The code commented below resulted in the time per epoch of
      # an 8-GPU wide resnet increasing by about 5% relative to the
      # code now in use.
      # The two different implementations have the same forward prop
      # down to machine precision on all inputs I have tested, but
      # sometimes have different derivatives.
      # Both obtain about the same training accuracy but the faster
      # version seems to also be slightly more accurate.
      # The commented code and these performance notes are included to
      # aid future revision efforts.
      #
      # out = out - self.leak * tf.nn.relu(-x)
      #

      out = tf.where(tf.less(x, 0.0), self.leak * x, x)
    return out

  def get_params(self):
    return []


class Sigmoid(Layer):

  def __init__(self, **kwargs):
    super(Sigmoid, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_output_shape(self):
    return self.output_shape

  def fprop(self, x, **kwargs):
    return tf.nn.sigmoid(x)

  def get_params(self):
    return []


class Tanh(Layer):

  def __init__(self, **kwargs):
    super(Tanh, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_output_shape(self):
    return self.output_shape

  def fprop(self, x, **kwargs):
    return tf.nn.tanh(x)

  def get_params(self):
    return []


class LeakyReLU(ReLU):

  def __init__(self, leak=.2, **kwargs):
    super(LeakyReLU, self).__init__(leak=leak, **kwargs)


class ELU(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_output_shape(self):
    return self.output_shape

  def fprop(self, x, **kwargs):
    return tf.nn.elu(x)

  def get_params(self):
    return []


class SELU(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_output_shape(self):
    return self.output_shape

  def fprop(self, x, **kwargs):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    mask = tf.to_float(x >= 0.)
    out = mask * x + (1. - mask) * \
        (alpha * tf.exp((1. - mask) * x) - alpha)
    return scale * out

  def get_params(self):
    return []


class TanH(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_output_shape(self):
    return self.output_shape

  def fprop(self, x, **kwargs):
    return tf.nn.tanh(x)


class Softmax(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def fprop(self, x, **kwargs):
    out = tf.nn.softmax(x)
    return out

  def get_params(self):
    return []

  def make_label_placeholder(self):
    return tf.placeholder(tf.float32, self.output_shape)


class Flatten(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    output_width = 1
    for factor in shape[1:]:
      output_width *= factor
    self.output_width = output_width
    self.output_shape = [None, output_width]

  def fprop(self, x, **kwargs):
    return tf.reshape(x, [-1, self.output_width])

  def get_params(self):
    return []


class Print(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_params(self):
    return []

  def fprop(self, x, **kwargs):
    mean = tf.reduce_mean(x)
    std = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
    return tf.Print(x,
                    [tf.reduce_min(x), mean, tf.reduce_max(x), std],
                    "Print layer")


class Add(Layer):
  """
  A Layer that adds a function to its input.
  The function to add is specified in terms of multiple layers, just like
  in the MLP class.
  The Add layer is useful for implementing residual networks.
  """

  def __hash__(self):
    return hash(id(self))

  def set_input_shape(self, shape):
    self.input_shape = shape
    shapes = {"input": shape}
    for layer in self.layers:
      layer.set_input_shape(shapes[layer.parent])
      shapes[layer.name] = layer.get_output_shape()
    self.output_shape = shapes[self.layers[-1].name]

  def __init__(self, layers):
    super(Add, self).__init__()

    self.layer_names = []
    self.layers = layers

    for i, layer in enumerate(self.layers):
      if layer.parent is None:
        if i == 0:
          layer.parent = "input"
        else:
          layer.parent = layers[i - 1].name
      if hasattr(layer, 'name'):
        name = layer.name
      else:
        name = layer.__class__.__name__ + str(i)
        layer.name = name
      self.layer_names.append(name)

  def get_params(self):
    out = []
    for layer in self.layers:
      out = ordered_union(out, layer.get_params())
    return out

  def fprop(self, x, **kwargs):

    orig_x = x

    # Note: this currently isn't great.
    # A layer can have any parent it wants, but the parent
    # must come earlier in the list.
    # There's no way to have > 1 parent.
    # This means we can support branched structures that split,
    # e.g. for multiple output heads, but not structures
    # that converge.
    # We can feed a value in the middle using "given" but
    # only layers after the given one are run using the current
    # implementation, so the feed must happen before any branch
    # point.

    out = {'input': x}

    for layer in self.layers:
      x = out[layer.parent]
      try:
        x = layer.fprop(x)
      except TypeError as e:
        msg = "TypeError in fprop for layer %s of type %s: %s"
        msg = msg % (layer.name, str(type(layer)), str(e))
        raise TypeError(msg)
      assert x is not None
      out[layer.name] = x

    return orig_x + out[self.layers[-1].name]


class PerImageStandardize(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_params(self):
    return []

  def fprop(self, x, **kwargs):
    axis = [1, 2, 3]
    mean = reduce_mean(x, axis=axis, keepdims=True)
    variance = reduce_mean(
        tf.square(x), axis=axis, keepdims=True) - tf.square(mean)
    variance = tf.nn.relu(variance)
    stddev = tf.sqrt(variance)

    num_pixels = reduce_prod(tf.shape(x)[1:])

    min_stddev = tf.rsqrt(tf.to_float(num_pixels))
    pixel_value_scale = tf.maximum(stddev, min_stddev)
    pixel_value_offset = mean

    x = tf.subtract(x, pixel_value_offset)
    x = tf.div(x, pixel_value_scale)
    return x


class Dropout(Layer):
  """Dropout layer.

  By default, is a no-op. Activate it during training using the kwargs

  The default use case is that you never specify include_prob anywhere.
  During evaluation, you don't do anything special regarding dropout,
  and nothing gets dropped. During training, you pass "dropout=True"
  to make units get randomly dropped.
  If you've done nothing else, include_prob defaults to 0.5 in the
  Dropout class constructor.

  A slightly more advanced use case is that you want to use different
  include_probs for some layer. For example, people usually use
  include_prob=0.8 for the input layer. To do this, you specify
  include_prob in the constructor arguments for those layers.
  Other than that, it's the same as the basic use case case. You do
  nothing special at test time and nothing is dropped.
  You pass dropout=True at train time and units in each layer are dropped
  based on their include_prob specified in their layer's constructor.

  The most advanced use case is if you want to change dropout include
  probabilities for a specific fprop call. In this case, you pass
  dropout=True and dropout_dict for that call. Each layer uses the
  include_prob specified in the dropout_dict for that call.of MLP.fprop.
  """

  def __init__(self, include_prob=0.5, **kwargs):
    super(Dropout, self).__init__(**kwargs)
    self.include_prob = include_prob

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def get_params(self):
    return []

  def fprop(self, x, dropout=False, dropout_dict=None, **kwargs):
    """
    Forward propagation as either no-op or dropping random units.
    :param x: The input to the layer
    :param dropout: bool specifying whether to drop units
    :param dropout_dict: dict
        This dictionary is usually not needed.
        In rare cases, generally for research purposes, this dictionary
        makes it possible to run forward propagation with a different
        dropout include probability.
        This dictionary should be passed as a named argument to the MLP
        class, which will then pass it to *all* layers' fprop methods.
        Other layers will just receive this as an ignored kwargs entry.
        Each dropout layer looks up its own name in this dictionary
        to read out its include probability.
    """
    include_prob = self.include_prob
    if dropout_dict is not None:
      assert dropout
      if self.name in dropout_dict:
        include_prob = dropout_dict[self.name]
    if dropout:
      return tf.nn.dropout(x, include_prob)
    return x


class ResidualWithGroupNorm(Layer):
  """A residual network layer that uses group normalization.

  :param out_filter: Number of output filters
  :param stride: int
      Stride for convolutional layers. Replicated to both row and column.
  """

  def __init__(self, out_filter, stride, activate_before_residual=False,
               leak=0.1, **kwargs):
    assert isinstance(stride, int)
    self.__dict__.update(locals())
    del self.self
    self.lrelu = LeakyReLU(leak)
    super(ResidualWithGroupNorm, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = tuple(shape)
    self.in_filter = shape[-1]
    self.gn1 = GroupNorm(name=self.name + "_gn1")
    self.gn1.set_input_shape(shape)
    strides = (self.stride, self.stride)
    self.conv1 = Conv2D(self.out_filter, (3, 3), strides, "SAME",
                        name=self.name + "_conv1", init_mode="inv_sqrt")
    self.conv1.set_input_shape(shape)
    self.gn2 = GroupNorm(name=self.name + "_gn2")
    self.gn2.set_input_shape(self.conv1.get_output_shape())
    self.conv2 = Conv2D(self.out_filter, (3, 3), (1, 1), "SAME",
                        name=self.name + "_conv2", init_mode="inv_sqrt")
    self.conv2.set_input_shape(self.conv1.get_output_shape())
    self.output_shape = self.conv2.get_output_shape()

  def get_params(self):
    sublayers = [self.conv1, self.conv2, self.gn1, self.gn2]
    params = []
    for sublayer in sublayers:
      params = params + sublayer.get_params()
    assert self.conv1.kernels.var in params
    return params

  def fprop(self, x, **kwargs):
    if self.activate_before_residual:
      x = self.gn1.fprop(x)
      x = self.lrelu.fprop(x)
      orig_x = x
    else:
      orig_x = x
      x = self.gn1.fprop(x)
      x = self.lrelu.fprop(x)
    x = self.conv1.fprop(x)
    x = self.gn2.fprop(x)
    x = self.lrelu.fprop(x)
    x = self.conv2.fprop(x)
    if self.stride != 1:
      stride = [1, self.stride, self.stride, 1]
      orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
    out_filter = self.out_filter
    in_filter = self.in_filter
    if in_filter != out_filter:
      orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                               [(out_filter - in_filter) // 2,
                                (out_filter - in_filter) // 2]])
    x = x + orig_x
    return x


class GlobalAveragePool(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = [shape[0], shape[-1]]

  def get_params(self):
    return []

  def fprop(self, x, **kwargs):
    assert len(list(x.get_shape())) == 4
    return tf.reduce_mean(x, [1, 2])


class GroupNorm(Layer):
  """
  Group normalization

  https://arxiv.org/abs/1803.08494
  """

  def __init__(self, num_groups=32, eps=1e-3, init_gamma=1.,
               **kwargs):
    self.num_groups = num_groups
    self.eps = eps
    self.init_gamma = init_gamma
    super(GroupNorm, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape
    channels = shape[-1]
    self.channels = channels
    self.actual_num_groups = min(self.channels, self.num_groups)
    extra_dims = (self.channels // self.actual_num_groups,
                  self.actual_num_groups)
    self.expanded_shape = tuple(shape[1:3]) + tuple(extra_dims)
    init_value = np.ones((channels,), dtype='float32') * self.init_gamma
    self.gamma = PV(init_value, name=self.name + "_gamma")
    self.beta = PV(np.zeros((self.channels,), dtype='float32'),
                   name=self.name + "_beta")

  def fprop(self, x, **kwargs):
    shape = tf.shape(x)
    batch_size = shape[0]
    x = tf.reshape(x, (batch_size,) + self.expanded_shape)
    mean, var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
    x = (x - mean) * tf.rsqrt(var + self.eps)
    x = tf.reshape(x, shape)
    x = x * self.gamma.var + self.beta.var
    return x

  def get_params(self):
    return [self.gamma.var, self.beta.var]


class BatchNorm(Layer):
  """
  This BatchNorm is always run in train mode for now
  """

  def __init__(self, eps=1e-3, init_gamma=1.,
               **kwargs):
    self.eps = eps
    self.init_gamma = init_gamma
    super(BatchNorm, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape
    channels = shape[-1]
    init_value = np.ones((channels,), dtype='float32') * self.init_gamma
    self.gamma = PV(init_value, name=self.name + "_gamma")
    self.beta = PV(np.zeros((channels,), dtype='float32'),
                   name=self.name + "_beta")

  def fprop(self, x, **kwargs):
    mean, var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    x = (x - mean) * tf.rsqrt(var + self.eps)
    x = x * self.gamma.var + self.beta.var
    return x

  def get_params(self):
    return [self.gamma.var, self.beta.var]


class ResidualWithBatchNorm(Layer):
  """A residual network layer that uses batch normalization.

  :param out_filter: Number of output filters
  :param stride: int
      Stride for convolutional layers. Replicated to both row and column.
  """

  def __init__(self, out_filter, stride, activate_before_residual=False,
               leak=0.1, **kwargs):
    assert isinstance(stride, int)
    self.__dict__.update(locals())
    del self.self
    self.lrelu = LeakyReLU(leak)
    super(ResidualWithBatchNorm, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = tuple(shape)
    self.in_filter = shape[-1]
    self.bn1 = BatchNorm(name=self.name + "_bn1")
    self.bn1.set_input_shape(shape)
    strides = (self.stride, self.stride)
    self.conv1 = Conv2D(self.out_filter, (3, 3), strides, "SAME",
                        name=self.name + "_conv1", init_mode="inv_sqrt")
    self.conv1.set_input_shape(shape)
    self.bn2 = BatchNorm(name=self.name + "_bn2")
    self.bn2.set_input_shape(self.conv1.get_output_shape())
    self.conv2 = Conv2D(self.out_filter, (3, 3), (1, 1), "SAME",
                        name=self.name + "_conv2", init_mode="inv_sqrt")
    self.conv2.set_input_shape(self.conv1.get_output_shape())
    self.output_shape = self.conv2.get_output_shape()

  def get_params(self):
    sublayers = [self.conv1, self.conv2, self.bn1, self.bn2]
    params = []
    for sublayer in sublayers:
      params = params + sublayer.get_params()
    assert self.conv1.kernels.var in params
    return params

  def fprop(self, x, **kwargs):
    if self.activate_before_residual:
      x = self.bn1.fprop(x)
      x = self.lrelu.fprop(x)
      orig_x = x
    else:
      orig_x = x
      x = self.bn1.fprop(x)
      x = self.lrelu.fprop(x)
    x = self.conv1.fprop(x)
    x = self.bn2.fprop(x)
    x = self.lrelu.fprop(x)
    x = self.conv2.fprop(x)
    if self.stride != 1:
      stride = [1, self.stride, self.stride, 1]
      orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
    out_filter = self.out_filter
    in_filter = self.in_filter
    if in_filter != out_filter:
      orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                               [(out_filter - in_filter) // 2,
                                (out_filter - in_filter) // 2]])
    x = x + orig_x
    return x
