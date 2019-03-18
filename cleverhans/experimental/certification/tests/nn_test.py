"""Tests for cleverhans.experimental.certification.nn."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf

from cleverhans.experimental.certification import nn


class NeuralNetworkTest(unittest.TestCase):

  def test_init(self):
    # Function to test initialization of NeuralNetParams object.
    # Valid params
    net_weights = [[[2, 2], [3, 3], [4, 4]], [1, 1, 1]]
    net_biases = [
        np.transpose(np.matrix([0, 0, 0])),
        np.transpose(np.matrix([0, 0]))
    ]
    net_layer_types = ['ff_relu', 'ff']
    nn_params1 = nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

    self.assertIsNotNone(nn_params1)
    # Invalid params : list length
    net_biases = [0]
    with self.assertRaises(ValueError):
      nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

    # Invalid params: layer types
    with self.assertRaises(ValueError):
      net_layer_types = ['ff_relu', 'ff_relu']
      nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

  def test_forward_pass(self):
    # Function to test forward pass of nn_params.
    net_weights = [[[2, 2], [3, 3], [4, 4]], [1, 1, 1]]
    net_biases = [
        np.transpose(np.matrix([0, 0, 0])),
        np.transpose(np.matrix([0, 0]))
    ]
    net_layer_types = ['ff_relu', 'ff']
    nn_params = nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

    input_vector = tf.random_uniform(shape=(2, 1), dtype=tf.float32)
    output_vector = nn_params.forward_pass(input_vector, 0)
    self.assertEqual(output_vector.shape.as_list(), [3, 1])
    output_vector_2 = nn_params.forward_pass(input_vector, 0, is_abs=True)
    self.assertEqual(output_vector_2.shape.as_list(), [3, 1])
    input_vector_trans = tf.random_uniform(shape=(3, 1), dtype=tf.float32)
    output_vector_3 = nn_params.forward_pass(
        input_vector_trans, 0, is_transpose=True)
    self.assertEqual(output_vector_3.shape.as_list(), [2, 1])


if __name__ == '__main__':
  unittest.main()
