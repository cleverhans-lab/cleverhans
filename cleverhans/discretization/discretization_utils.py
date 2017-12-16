"""Utilty functions for discretizing image tensors in various ways.

For the discretization, we may either use uniform buckets or supply our own
custom buckets. One way to compute custom buckets is to use percentile
information from the data distribution. The final discretized representation
can either be a one-hot or a thermometer encoding. A thermometer encoding
is of the form (1, 1, 1,..,1, 0, .., 0) with the transition from 1 to 0
signifying which bucket it belongs to. To reduce the dimension, one may
project back by convolving with a fixed random or trainable matrix.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def flatten_last(x):
  """Flatten the last two dimensions of a tensor into one.

  Args:
    x: Discretized input tensor of shape [-1, height, width, channels, levels]
        to flatten.

  Returns:
    Flattened version of x, of shape [-1, height, width, channels * levels].
  """
  shape = x.get_shape().as_list()
  new_shape = shape[:-1]
  new_shape[-1] *= shape[-1]
  new_shape[0] = tf.shape(x)[0]
  flattened_x = tf.reshape(x, new_shape)
  return flattened_x


def unflatten_last(x, levels):
  """Unflatten input tensor by separating the last two dimensions.

  Args:
    x: Discretized input tensor of shape [-1, height, width, channels * levels]
        to unflatten.
    levels: Number of levels the tensor has been discretized into.

  Returns:
    Unflattened version of x, of shape [-1, height, width, channels, levels].
  """
  shape = x.get_shape().as_list()
  shape[-1] /= levels
  shape[-1] = int(shape[-1])
  shape.append(levels)
  shape[0] = tf.shape(x)[0]
  unflattened_x = tf.reshape(x, shape)
  return unflattened_x


def discretize_uniform(x, levels, thermometer=False):
  """Discretize input into levels using uniformly distributed buckets.

  Args:
    x: Input tensor to discretize, assumed to be between (0, 1).
    levels: Number of levels to discretize into.
    thermometer: Whether to encode the discretized tensor in thermometer encoding
        (Default: False).


  Returns:
    Discretized version of x of shape [-1, height, width, channels * levels].
  """
  clipped_x = tf.clip_by_value(x, 0., 1.)
  int_x = tf.to_int32((.99999 * clipped_x) * levels)
  one_hot = tf.one_hot(
      int_x, depth=levels, on_value=1., off_value=0., dtype=tf.float32, axis=-1)

  # Check to see if we are encoding in thermometer
  discretized_x = one_hot
  if thermometer:
    discretized_x = one_hot_to_thermometer(one_hot, levels, flattened=False)

  # Reshape x to [-1, height, width, channels * levels]
  discretized_x = flatten_last(discretized_x)
  return discretized_x


def get_centroids_by_percentile(x, levels):
  """Get the custom centroids by percentiles of the per-pixel distribution of x.

  Args:
    x: Input data set of shape [-1, height, width, channels]
        whose centroids we wish to compute.
    levels: Number of centroids to compute.

  Returns:
    Custom centroids as a tensor.
  """

  def quantile(q):
    return tf.contrib.distributions.percentile(x, q=q, axis=None)

  start = 0.
  end = 100.
  quantile_range = tf.lin_space(start, end, levels)
  centroids = tf.map_fn(quantile, quantile_range)
  return centroids


def discretize_centroids(x, levels, centroids, thermometer=False):
  """Discretize input into levels using custom centroids.

  Args:
    x: Input tensor to discretize, assumed to be between (0, 1).
    levels: Number of levels to discretize into.
    centroids: Custom centroids into which the input is to be discretized.
    thermometer: Whether to encode the discretized tensor in thermometer encoding
        (Default: False).

  Returns:
    Discretized version of x of shape [-1, height, width, channels * levels]
    using supplied centroids.
  """
  x_stacked = tf.stack(levels * [x], axis=-1)
  dist = tf.to_float(tf.squared_difference(x_stacked, centroids))
  idx = tf.argmin(dist, axis=-1)
  one_hot = tf.one_hot(idx, depth=levels, on_value=1., off_value=0.)

  # Check to see if we are encoding in thermometer
  discretized_x = one_hot
  if thermometer:
    discretized_x = one_hot_to_thermometer(one_hot, levels, flattened=False)

  # Reshape x to [-1, height, width, channels * levels]
  discretized_x = flatten_last(discretized_x)
  return discretized_x


def undiscretize_uniform(x, levels, flattened=False, thermometer=False):
  """Undiscretize a discretized tensor.

  Args:
    x: Input tensor in discretized form.
    levels: Number of levels the input has been discretized into.
    flattened: True if x is of the form [-1, height, width, channels * levels]
        else it is of shape [-1, height, width, channels, levels].
        (Default: False).
    thermometer: Determines if we are using one-hot or thermometer encoding
        (Default: False).

  Returns:
    Undiscretized version of x.
  """
  # Unflatten if flattened, so that x has shape
  # [-1, height, width, channels, levels]
  if flattened:
    x = unflatten_last(x, levels)
  if thermometer:
    int_x = tf.reduce_sum(x, -1) - 1
  else:
    int_x = tf.argmax(x, -1)
  out = tf.to_float(int_x) / (levels - 1)
  return out


def undiscretize_centroids(x,
                           levels,
                           centroids,
                           flattened=False,
                           thermometer=False):
  """Undiscretize a tensor that has been discretized using custom centroids.

  Args:
    x: Input tensor in discretized form.
    levels: Number of levels the input has been discretized into.
    centroids: The custom centroids used to discretize.
    flattened: True if x is of the form [-1, height, width, channels * levels]
        else it is of shape [-1, height, width, channels, levels].
        (Default: False).
    thermometer: Determines if we are using one-hot or thermometer encoding
        (Default: False).

  Returns:
    Undiscretized version of x.
  """
  # Unflatten if flattened, so that x has shape
  # [-1, height, width, channels, levels]
  if flattened:
    x = unflatten_last(x, levels)
  if thermometer:
    x = thermometer_to_one_hot(x, levels, flattened=False)
  out = tf.reduce_sum(tf.multiply(x, centroids), axis=-1)
  return out


def one_hot_to_thermometer(x, levels, flattened=False):
  """Convert one hot to thermometer code.

  Args:
    x: Input tensor in one hot encoding to convert to thermometer.
    levels: Number of levels the input has been discretized into.
    flattened: True if x is of the form [-1, height, width, channels * levels]
        else it is of shape [-1, height, width, channels, levels].
        (Default: False).

  Returns:
    Thermometer encoding of x.
  """
  # Unflatten if flattened, so that x has shape
  # [-1, height, width, channels, levels]
  if flattened:
    x = unflatten_last(x, levels)
  thermometer = tf.cumsum(x, axis=-1, reverse=True)
  # Flatten back if original input was flattened
  if flattened:
    thermometer = flatten_last(thermometer)
  return thermometer


def thermometer_to_one_hot(x, levels, flattened=False):
  """Convert thermometer to one hot code.

  Args:
    x: Input tensor in thermometer encoding to convert to one-hot. Input is
        assumed to be
        of shape [-1, height, width, channels, levels].
    levels: Number of levels the input has been discretized into.
    flattened: True if x is of the form [-1, height, width, channels * levels]
        else it is of shape [-1, height, width, channels, levels].
        (Default: False).

  Returns:
    One hot encoding of x.
  """
  # Unflatten if flattened, so that x has shape
  # [-1, height, width, channels, levels]
  if flattened:
    x = unflatten_last(x, levels)
  int_x = tf.to_int32(tf.reduce_sum(x, axis=-1)) - 1
  one_hot = tf.one_hot(
      int_x, depth=levels, on_value=1., off_value=0., dtype=tf.float32, axis=-1)
  # Flatten back if input was flattened
  if flattened:
    one_hot = flatten_last(one_hot)
  return one_hot


def random_convolution(x,
                       projection_dim,
                       levels,
                       flattened=True,
                       trainable=False):
  """Reduce dimension by random convolutions using a standard Gaussian.

  Args:
    x: Discretized input tensor in one hot or thermometer encoding to project.
    projection_dim: Dimension to project the output tensor to.
    levels: Number of levels the input has been discretized into.
    flattened: True if x is of the form [-1, height, width, channels * levels]
        else it is of shape [-1, height, width, channels * levels].
        (Default: False).
    trainable: If True then the weights for projection are learned (Default:
        False).

  Returns:
    Projection of x using a fixed random convolution.

  Raises:
    ValueError: If projection dimension is higher than the number of levels.
  """
  if projection_dim > levels:
    raise ValueError('Projection dimension higher than the number of levels')

  # Unflatten first to get number of channels
  if flattened:
    x = unflatten_last(x, levels)

  channels = x.get_shape().as_list()[3]

  # Flatten so that x has shape [-1, height, width, channels * levels]
  x = flatten_last(x)

  scope = 'projection'
  if trainable:
    scope = 'trainable_projection'

  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    kernel = tf.get_variable(
        'conv_projection', [1, 1, channels * levels, channels * projection_dim],
        trainable=trainable)

  x_proj = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

  # Unflatten back if input was not flattened
  if not flattened:
    x_proj = unflatten_last(x_proj, levels)
  return x_proj

