# pylint: disable=missing-docstring
import functools
import tensorflow as tf

from cleverhans.initializers import HeReLuNormalInitializer
from cleverhans.model import Model


class ModelImageNetCNN(Model):
  def __init__(self, scope, nb_classes=1000, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())

  def fprop(self, x, **kwargs):
    del kwargs
    my_conv = functools.partial(tf.layers.conv2d,
                                kernel_size=3,
                                strides=2,
                                padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=HeReLuNormalInitializer)
    my_dense = functools.partial(
        tf.layers.dense, kernel_initializer=HeReLuNormalInitializer)

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      for depth in [96, 256, 384, 384, 256]:
        x = my_conv(x, depth)
      y = tf.layers.flatten(x)
      y = my_dense(y, 4096, tf.nn.relu)
      y = fc7 = my_dense(y, 4096, tf.nn.relu)
      y = my_dense(y, 1000)
      return {'fc7': fc7,
              self.O_LOGITS: y,
              self.O_PROBS: tf.nn.softmax(logits=y)}


def make_imagenet_cnn(input_shape=(None, 224, 224, 3)):
  return ModelImageNetCNN('imagenet')
