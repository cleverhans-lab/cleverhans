"""Tests for cleverhans.experimental.certification.dual_formulation."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf

from cleverhans.experimental.certification import dual_formulation
from cleverhans.experimental.certification import nn


class DualFormulationTest(unittest.TestCase):

  def test_init(self):
    # Function to test initialization of dual formulation class.
    net_weights = [[[2, 2], [3, 3], [4, 4]], [[1, 1, 1], [-1, -1, -1]]]
    net_biases = [np.transpose(np.matrix([0, 0, 0])),
                  np.transpose(np.matrix([0, 0]))]
    net_layer_types = ['ff_relu', 'ff']
    nn_params1 = nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

    test_input = np.transpose(np.matrix([0, 0]))
    true_class = 0
    adv_class = 1
    input_minval = 0
    input_maxval = 0
    epsilon = 0.1
    three_dim_tensor = tf.random_uniform(shape=(3, 1), dtype=tf.float32)
    two_dim_tensor = tf.random_uniform(shape=(2, 1), dtype=tf.float32)
    scalar = tf.random_uniform(shape=(1, 1), dtype=tf.float32)
    lambda_pos = [two_dim_tensor, three_dim_tensor]
    lambda_neg = lambda_pos
    lambda_quad = lambda_pos
    lambda_lu = lambda_pos
    nu = scalar
    dual_var = {
        'lambda_pos': lambda_pos,
        'lambda_neg': lambda_neg,
        'lambda_quad': lambda_quad,
        'lambda_lu': lambda_lu,
        'nu': nu
    }
    with tf.Session() as sess:
      dual_formulation_object = dual_formulation.DualFormulation(sess,
                                                                 dual_var,
                                                                 nn_params1,
                                                                 test_input,
                                                                 true_class,
                                                                 adv_class,
                                                                 input_minval,
                                                                 input_maxval,
                                                                 epsilon)
    self.assertIsNotNone(dual_formulation_object)

  def test_set_differentiable_objective(self):
    # Function to test the function that sets the differentiable objective.
    net_weights = [[[2, 2], [3, 3], [4, 4]], [[1, 1, 1], [-1, -1, -1]]]
    net_biases = [
        np.transpose(np.matrix([0, 0, 0])),
        np.transpose(np.matrix([0, 0]))
    ]
    net_layer_types = ['ff_relu', 'ff']
    nn_params1 = nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

    test_input = np.transpose(np.matrix([0, 0]))
    true_class = 0
    adv_class = 1
    input_minval = 0
    input_maxval = 0
    epsilon = 0.1
    three_dim_tensor = tf.random_uniform(shape=(3, 1), dtype=tf.float32)
    two_dim_tensor = tf.random_uniform(shape=(2, 1), dtype=tf.float32)
    scalar = tf.random_uniform(shape=(1, 1), dtype=tf.float32)
    lambda_pos = [two_dim_tensor, three_dim_tensor]
    lambda_neg = lambda_pos
    lambda_quad = lambda_pos
    lambda_lu = lambda_pos
    nu = scalar
    dual_var = {
        'lambda_pos': lambda_pos,
        'lambda_neg': lambda_neg,
        'lambda_quad': lambda_quad,
        'lambda_lu': lambda_lu,
        'nu': nu
    }
    with tf.Session() as sess:
      dual_formulation_object = dual_formulation.DualFormulation(sess,
                                                                 dual_var,
                                                                 nn_params1,
                                                                 test_input,
                                                                 true_class,
                                                                 adv_class,
                                                                 input_minval,
                                                                 input_maxval,
                                                                 epsilon)
    dual_formulation_object.set_differentiable_objective()
    self.assertEqual(dual_formulation_object.scalar_f.shape.as_list(), [1])
    self.assertEqual(
        dual_formulation_object.unconstrained_objective.shape.as_list(), [1, 1])
    self.assertEqual(dual_formulation_object.vector_g.shape.as_list(), [5, 1])

  def test_get_full_psd_matrix(self):
    # Function to test product with PSD matrix.
    net_weights = [[[2, 2], [3, 3], [4, 4]], [[1, 1, 1], [-1, -1, -1]]]
    net_biases = [
        np.transpose(np.matrix([0, 0, 0])),
        np.transpose(np.matrix([0, 0]))
    ]
    net_layer_types = ['ff_relu', 'ff']
    nn_params1 = nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

    test_input = np.transpose(np.matrix([0, 0]))
    true_class = 0
    adv_class = 1
    input_minval = 0
    input_maxval = 0
    epsilon = 0.1
    three_dim_tensor = tf.random_uniform(shape=(3, 1), dtype=tf.float32)
    two_dim_tensor = tf.random_uniform(shape=(2, 1), dtype=tf.float32)
    scalar = tf.random_uniform(shape=(1, 1), dtype=tf.float32)
    lambda_pos = [two_dim_tensor, three_dim_tensor]
    lambda_neg = lambda_pos
    lambda_quad = lambda_pos
    lambda_lu = lambda_pos
    nu = scalar
    dual_var = {
        'lambda_pos': lambda_pos,
        'lambda_neg': lambda_neg,
        'lambda_quad': lambda_quad,
        'lambda_lu': lambda_lu,
        'nu': nu
    }
    with tf.Session() as sess:
      dual_formulation_object = dual_formulation.DualFormulation(sess,
                                                                 dual_var,
                                                                 nn_params1,
                                                                 test_input,
                                                                 true_class,
                                                                 adv_class,
                                                                 input_minval,
                                                                 input_maxval,
                                                                 epsilon)
    matrix_h, matrix_m = dual_formulation_object.get_full_psd_matrix()
    self.assertEqual(matrix_h.shape.as_list(), [5, 5])
    self.assertEqual(matrix_m.shape.as_list(), [6, 6])

  def test_get_psd_product(self):
    # Function to test implicit product with PSD matrix.
    net_weights = [[[2, 2], [3, 3], [4, 4]], [[1, 1, 1], [-1, -1, -1]]]
    net_biases = [
        np.transpose(np.matrix([0, 0, 0])),
        np.transpose(np.matrix([0, 0]))
    ]
    net_layer_types = ['ff_relu', 'ff']
    nn_params1 = nn.NeuralNetwork(net_weights, net_biases, net_layer_types)

    test_input = np.transpose(np.matrix([0, 0]))
    true_class = 0
    adv_class = 1
    input_minval = 0
    input_maxval = 0
    epsilon = 0.1
    three_dim_tensor = tf.random_uniform(shape=(3, 1), dtype=tf.float32)
    two_dim_tensor = tf.random_uniform(shape=(2, 1), dtype=tf.float32)
    scalar = tf.random_uniform(shape=(1, 1), dtype=tf.float32)
    lambda_pos = [two_dim_tensor, three_dim_tensor]
    lambda_neg = lambda_pos
    lambda_quad = lambda_pos
    lambda_lu = lambda_pos
    nu = scalar
    dual_var = {
        'lambda_pos': lambda_pos,
        'lambda_neg': lambda_neg,
        'lambda_quad': lambda_quad,
        'lambda_lu': lambda_lu,
        'nu': nu
    }
    with tf.Session() as sess:
      dual_formulation_object = dual_formulation.DualFormulation(sess,
                                                                 dual_var,
                                                                 nn_params1,
                                                                 test_input,
                                                                 true_class,
                                                                 adv_class,
                                                                 input_minval,
                                                                 input_maxval,
                                                                 epsilon)
      _, matrix_m = dual_formulation_object.get_full_psd_matrix()

      # Testing if the values match
      six_dim_tensor = tf.random_uniform(shape=(6, 1), dtype=tf.float32)
      implicit_product = dual_formulation_object.get_psd_product(six_dim_tensor)
      explicit_product = tf.matmul(matrix_m, six_dim_tensor)
      [implicit_product_value,
       explicit_product_value] = sess.run([implicit_product, explicit_product])
      self.assertEqual(
          np.shape(implicit_product_value), np.shape(explicit_product_value))
      self.assertLess(
          np.max(np.abs(implicit_product_value - explicit_product_value)), 1E-5)


if __name__ == '__main__':
  unittest.main()
