"""cleverhans.model.Model implementation of mnist_challenge.model.Model

This re-implementation factors variable creation apart from forward
propagation so it is possible to run forward propagation more than once
in the same model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf
from cleverhans.model import Model
from cleverhans.utils import deterministic_dict
from cleverhans.dataset import Factory, MNIST


class MadryMNIST(Model):

  def __init__(self, nb_classes=10):
    # NOTE: for compatibility with Madry Lab downloadable checkpoints,
    # we cannot use scopes, give these variables names, etc.
    self.W_conv1 = self._weight_variable([5, 5, 1, 32])
    self.b_conv1 = self._bias_variable([32])
    self.W_conv2 = self._weight_variable([5, 5, 32, 64])
    self.b_conv2 = self._bias_variable([64])
    self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    self.b_fc1 = self._bias_variable([1024])
    self.W_fc2 = self._weight_variable([1024, nb_classes])
    self.b_fc2 = self._bias_variable([nb_classes])
    Model.__init__(self, '', nb_classes, {})
    self.dataset_factory = Factory(MNIST, {"center": False})

  def get_params(self):
    return [
        self.W_conv1,
        self.b_conv1,
        self.W_conv2,
        self.b_conv2,
        self.W_fc1,
        self.b_fc1,
        self.W_fc2,
        self.b_fc2,
    ]

  def fprop(self, x):

    output = OrderedDict()
    # first convolutional layer
    h_conv1 = tf.nn.relu(self._conv2d(x, self.W_conv1) + self.b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    h_conv2 = tf.nn.relu(
        self._conv2d(h_pool1, self.W_conv2) + self.b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

    # output layer
    logits = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

    output = deterministic_dict(locals())
    del output["self"]
    output[self.O_PROBS] = tf.nn.softmax(logits=logits)

    return output

  @staticmethod
  def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  @staticmethod
  def _max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
