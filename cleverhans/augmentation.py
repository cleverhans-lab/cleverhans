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


def random_crop_and_flip(x, pad_rows=4, pad_cols=4):
  rows = tf.shape(x)[1]
  cols = tf.shape(x)[2]

  def _pad_image(img):
    return tf.image.resize_image_with_crop_or_pad(img, rows + pad_rows,
                                                      cols + pad_cols)

  def _random_crop_image(img):
    channels = img.get_shape()[2]
    return tf.random_crop(img, [rows, cols, channels])

  def random_crop_and_flip_image(img):
    return tf.image.random_flip_left_right(_random_crop_image(_pad_image(img)))

  # Some of these ops are only on CPU.
  # This function will often be called with the device set to GPU.
  # We need to set it to CPU temporarily to avoid an exception.
  with tf.device('/CPU:0'):
    x = tf.map_fn(random_crop_and_flip_image, x)
    # x = tf.map_fn(_pad_image, x)
    # x = tf.map_fn(_random_crop_image, x)
    # x = tf.map_fn(tf.image.random_flip_left_right, x)
  return x
