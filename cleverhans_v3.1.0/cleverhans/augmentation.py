"""
Dataset augmentation functionality

NOTE: This module is much more free to change than many other modules
in CleverHans. CleverHans is very conservative about changes to any
code that affects the output of benchmark tests (attacks, evaluation
methods, etc.). This module provides *dataset augmentation* code for
building models to be benchmarked, not *benchmarks,* and
thus is free to change rapidly to provide better speed, accuracy,
etc.
"""

import tensorflow as tf

# Convenient renaming of existing function
random_horizontal_flip = tf.image.random_flip_left_right


def random_shift(x, pad=(4, 4), mode='REFLECT'):
  """Pad a single image and then crop to the original size with a random
  offset."""
  assert mode in 'REFLECT SYMMETRIC CONSTANT'.split()
  assert x.get_shape().ndims == 3
  xp = tf.pad(x, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode)
  return tf.random_crop(xp, tf.shape(x))


def batch_augment(x, func, device='/CPU:0'):
  """
  Apply dataset augmentation to a batch of exmaples.
  :param x: Tensor representing a batch of examples.
  :param func: Callable implementing dataset augmentation, operating on
    a single image.
  :param device: String specifying which device to use.
  """
  with tf.device(device):
    return tf.map_fn(func, x)


def random_crop_and_flip(x, pad_rows=4, pad_cols=4):
  """Augment a batch by randomly cropping and horizontally flipping it."""
  rows = tf.shape(x)[1]
  cols = tf.shape(x)[2]
  channels = x.get_shape()[3]

  def _rand_crop_img(img):
    """Randomly crop an individual image"""
    return tf.random_crop(img, [rows, cols, channels])

  # Some of these ops are only on CPU.
  # This function will often be called with the device set to GPU.
  # We need to set it to CPU temporarily to avoid an exception.
  with tf.device('/CPU:0'):
    x = tf.image.resize_image_with_crop_or_pad(x, rows + pad_rows,
                                               cols + pad_cols)
    x = tf.map_fn(_rand_crop_img, x)
    x = tf.image.random_flip_left_right(x)
  return x
