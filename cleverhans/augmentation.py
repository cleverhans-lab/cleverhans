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


def random_horizontal_flip(x):
    return tf.image.random_flip_left_right(x)


def random_shift(x, pad=(4, 4), mode='REFLECT'):
    assert mode in 'REFLECT SYMMETRIC CONSTANT'.split()
    xp = tf.pad(x, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode)
    return tf.random_crop(xp, tf.shape(x))


def batch_augment(x, func, device='/CPU:0'):
    with tf.device(device):
        return tf.map_fn(func, x)


def random_crop_and_flip(x, pad_rows=4, pad_cols=4):
    rows = tf.shape(x)[1]
    cols = tf.shape(x)[2]

    def _pad_img(img):
        return tf.image.resize_image_with_crop_or_pad(img, rows + pad_rows,
                                                      cols + pad_cols)

    def _rand_crop_img(img):
        channels = img.get_shape()[2]
        return tf.random_crop(img, [rows, cols, channels])

    def random_crop_and_flip_image(img):
        return tf.image.random_flip_left_right(_rand_crop_img(_pad_img(img)))

    # Some of these ops are only on CPU.
    # This function will often be called with the device set to GPU.
    # We need to set it to CPU temporarily to avoid an exception.
    with tf.device('/CPU:0'):
        x = tf.map_fn(random_crop_and_flip_image, x)
    return x
