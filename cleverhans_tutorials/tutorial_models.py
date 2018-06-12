"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from cleverhans.model import Model


class ModelBasicCNN(Model):
    def __init__(self, scope, nb_classes, nb_filters, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = tf.layers.conv2d(x, self.nb_filters, 8, strides=2,
                                 padding='same', activation=tf.nn.relu)
            y = tf.layers.conv2d(y, self.nb_filters, 6, strides=2,
                                 padding='valid', activation=tf.nn.relu)
            y = tf.layers.conv2d(y, self.nb_filters, 5, strides=1,
                                 padding='valid', activation=tf.nn.relu)
            logits = tf.layers.dense(tf.layers.flatten(y), self.nb_classes)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}
