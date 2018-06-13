"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf
from cleverhans.model import Model


class ModelBasicCNN(Model):
    def __init__(self, scope, nb_classes, nb_filters, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters

    def fprop(self, x, **kwargs):
        del kwargs
        my_conv = functools.partial(tf.layers.conv2d, activation=tf.nn.relu,
                                    kernel_initializer=HeReLuNormalInitializer)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = my_conv(x, self.nb_filters, 8, strides=2, padding='same')
            y = my_conv(y, 2 * self.nb_filters, 6, strides=2, padding='valid')
            y = my_conv(y, 2 * self.nb_filters, 5, strides=1, padding='valid')
            logits = tf.layers.dense(tf.layers.flatten(y), self.nb_classes,
                                     kernel_initializer=HeReLuNormalInitializer)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


class HeReLuNormalInitializer(tf.initializers.random_normal):
    def __init__(self, dtype=tf.float32):
        self.dtype = tf.as_dtype(dtype)

    def get_config(self):
        return dict(dtype=self.dtype.name)

    def __call__(self, shape, dtype=None, partition_info=None):
        del partition_info
        dtype = self.dtype if dtype is None else dtype
        std = tf.rsqrt(tf.cast(tf.reduce_prod(shape[:-1]), tf.float32) + 1e-7)
        return tf.random_normal(shape, stddev=std, dtype=dtype)
