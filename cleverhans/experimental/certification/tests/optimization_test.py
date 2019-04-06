"""Tests for cleverhans.experimental.certification.optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from cleverhans.experimental.certification import dual_formulation
from cleverhans.experimental.certification import nn
from cleverhans.experimental.certification import optimization


class OptimizationTest(tf.test.TestCase):
  # pylint: disable=missing-docstring

  def prepare_dual_object(self):
    # Function to prepare dual object to be used for testing optimization.
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

    # Creating dual variables to use for optimization
    lambda_pos = [tf.get_variable('lambda_pos0',
                                  initializer=np.random.uniform(
                                      0, 0.1, size=(2, 1)).astype(np.float32)),
                  tf.get_variable('lambda_pos1',
                                  initializer=np.random.uniform(
                                      0, 0.1, size=(3, 1)).astype(np.float32))]
    lambda_neg = [tf.get_variable('lambda_neg0',
                                  initializer=np.random.uniform(
                                      0, 0.1, size=(2, 1)).astype(np.float32)),
                  tf.get_variable('lambda_neg1',
                                  initializer=np.random.uniform(
                                      0, 0.1, size=(3, 1)).astype(np.float32))]
    lambda_quad = [tf.get_variable('lambda_quad0',
                                   initializer=np.random.uniform(
                                       0, 0.1, size=(2, 1)).astype(np.float32)),
                   tf.get_variable('lambda_quad1',
                                   initializer=np.random.uniform(
                                       0, 0.1, size=(3, 1)).astype(np.float32))]
    lambda_lu = [tf.get_variable('lambda_lu0',
                                 initializer=np.random.uniform(
                                     0, 0.1, size=(2, 1)).astype(np.float32)),
                 tf.get_variable('lambda_lu1',
                                 initializer=np.random.uniform(
                                     0, 0.1, size=(3, 1)).astype(np.float32))]
    nu = tf.reshape(tf.get_variable('nu', initializer=200.0,
                                    dtype=tf.float32), shape=(1, 1))
    dual_var = {'lambda_pos': lambda_pos, 'lambda_neg': lambda_neg,
                'lambda_quad': lambda_quad, 'lambda_lu': lambda_lu, 'nu': nu}
    sess = tf.Session()
    dual_formulation_object = dual_formulation.DualFormulation(sess,
                                                               dual_var,
                                                               nn_params1,
                                                               test_input,
                                                               true_class,
                                                               adv_class,
                                                               input_minval,
                                                               input_maxval,
                                                               epsilon)
    return sess, dual_formulation_object

  def test_init(self):
    """ Function to test initialization of OptimizationTest. """
    sess, dual_formulation_object = self.prepare_dual_object()
    dual_formulation_object.set_differentiable_objective()
    sess.run(tf.global_variables_initializer())
    optimization_params = {
        'init_learning_rate': 0.1,
        'learning_rate_decay': 0.9,
        'eig_num_iter': 10,
        'eig_learning_rate': 0.01,
        'init_smooth': 0.5,
        'smooth_decay': 0.9,
        'inner_num_steps': 10,
        'optimizer': 'adam',
        'momentum_parameter': 0.9,
        'eig_type': 'TF'
    }
    optimization_object = optimization.Optimization(dual_formulation_object,
                                                    sess, optimization_params)
    self.assertIsNotNone(optimization_object)

  def test_get_min_eig_vec_proxy(self):
    """ Function test computing min eigen value using matrix vector products."""
    sess, dual_formulation_object = self.prepare_dual_object()
    _, matrix_m = dual_formulation_object.get_full_psd_matrix()
    optimization_params = {
        'init_learning_rate': 0.1,
        'learning_rate_decay': 0.9,
        'eig_num_iter': 2000,
        'eig_learning_rate': 0.01,
        'init_smooth': 0.0,
        'smooth_decay': 0.9,
        'inner_num_steps': 10,
        'optimizer': 'adam',
        'momentum_parameter': 0.9,
        'eig_type': 'TF'
    }
    sess.run(tf.global_variables_initializer())
    optimization_object = optimization.Optimization(dual_formulation_object,
                                                    sess, optimization_params)
    eig_vec = optimization_object.get_min_eig_vec_proxy()
    tf_eig_vec = optimization_object.get_min_eig_vec_proxy(use_tf_eig=True)
    self.assertIsNotNone(eig_vec)

    # Running the graphs and checking that minimum eigen value is correct
    # ** No smoothing
    tf_eig_vec_val, eig_vec_val, matrix_m_val = sess.run(
        [tf_eig_vec, eig_vec, matrix_m],
        feed_dict={
            optimization_object.eig_init_vec_placeholder:
                np.random.rand(6, 1),
            optimization_object.eig_num_iter_placeholder:
                2000,
            optimization_object.smooth_placeholder:
                0.0
        })

    # Eigen value corresponding to v is v^\top M v
    eig_val = np.matmul(
        np.transpose(eig_vec_val), np.matmul(matrix_m_val, eig_vec_val))
    tf_eig_val = np.matmul(
        np.transpose(tf_eig_vec_val), np.matmul(matrix_m_val, tf_eig_vec_val))
    [np_eig_values, _] = np.linalg.eig(matrix_m_val)
    self.assertLess(np.abs(np.min(np_eig_values) - eig_val), 1E-5)
    self.assertLess(np.abs(np.min(np_eig_values) - tf_eig_val), 1E-5)

    # Running the graphs and checking that minimum eigen value is correct
    # **Smoothing
    optimization_params['init_smooth'] = 0.0001
    optimization_object = optimization.Optimization(dual_formulation_object,
                                                    sess, optimization_params)
    eig_vec = optimization_object.get_min_eig_vec_proxy()
    tf_eig_vec = optimization_object.get_min_eig_vec_proxy(use_tf_eig=True)

    tf_eig_vec_val, eig_vec_val, matrix_m_val = sess.run(
        [tf_eig_vec, eig_vec, matrix_m],
        feed_dict={
            optimization_object.eig_init_vec_placeholder:
                np.random.rand(6, 1),
            optimization_object.smooth_placeholder:
                0.1,
            optimization_object.eig_num_iter_placeholder:
                2000
        })

    # Eigen value corresponding to v is v^\top M v
    eig_val = np.matmul(
        np.transpose(eig_vec_val), np.matmul(matrix_m_val, eig_vec_val))
    tf_eig_val = np.matmul(
        np.transpose(tf_eig_vec_val), np.matmul(matrix_m_val, tf_eig_vec_val))
    [np_eig_values, _] = np.linalg.eig(matrix_m_val)
    self.assertLess(np.abs(np.min(np_eig_values) - eig_val), 1E-5)
    # In general, smoothed version can be far off
    self.assertLess(np.abs(np.min(np_eig_values) - tf_eig_val), 1E-1)

  def test_optimization(self):
    """Function to test optimization."""
    sess, dual_formulation_object = self.prepare_dual_object()
    optimization_params = {
        'init_penalty': 10000,
        'large_eig_num_steps': 1000,
        'small_eig_num_steps': 500,
        'inner_num_steps': 10,
        'outer_num_steps': 2,
        'beta': 2,
        'smoothness_parameter': 0.001,
        'eig_learning_rate': 0.01,
        'optimizer': 'adam',
        'init_learning_rate': 0.1,
        'learning_rate_decay': 0.9,
        'momentum_parameter': 0.9,
        'print_stats_steps': 1,
        'stats_folder': None,
        'projection_steps': 200,
        'eig_type': 'TF'
    }
    sess.run(tf.global_variables_initializer())
    optimization_object = optimization.Optimization(dual_formulation_object,
                                                    sess, optimization_params)
    is_cert_found = optimization_object.run_optimization()
    self.assertFalse(is_cert_found)


if __name__ == '__main__':
  tf.test.main()
