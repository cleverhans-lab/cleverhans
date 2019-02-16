"""Extremely simple model where all parameters are from convolutions.
"""

import math
import tensorflow as tf

from cleverhans import initializers
from cleverhans.serial import NoRefModel


class ModelAllConvolutional(NoRefModel):
  """
  A simple model that uses only convolution and downsampling---no batch norm or other techniques that can complicate
  adversarial training.
  """
  def __init__(self, scope, nb_classes, nb_filters, input_shape, **kwargs):
    del kwargs
    NoRefModel.__init__(self, scope, nb_classes, locals())
    self.nb_filters = nb_filters
    self.input_shape = input_shape

    # Do a dummy run of fprop to create the variables from the start
    self.fprop(tf.placeholder(tf.float32, [32] + input_shape))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    conv_args = dict(
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializers.HeReLuNormalInitializer,
        kernel_size=3,
        padding='same')
    y = x

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      log_resolution = int(round(
          math.log(self.input_shape[0]) / math.log(2)))
      for scale in range(log_resolution - 2):
        y = tf.layers.conv2d(y, self.nb_filters << scale, **conv_args)
        y = tf.layers.conv2d(y, self.nb_filters << (scale + 1), **conv_args)
        y = tf.layers.average_pooling2d(y, 2, 2)
      y = tf.layers.conv2d(y, self.nb_classes, **conv_args)
      logits = tf.reduce_mean(y, [1, 2])
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}
