"""File containing some simple helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def diag(diag_elements):
  """Function to create tensorflow diagonal matrix with input diagonal entries.

  Args:
    diag_elements: tensor with diagonal elements

  Returns:
    tf matrix with diagonal entries as diag_elements
  """
  return tf.diag(tf.reshape(diag_elements, [-1]))


def initialize_dual(neural_net_params_object, init_dual_file=None,
                    random_init_variance=0.01, init_nu=200.0):
  """Function to initialize the dual variables of the class.

  Args:
    neural_net_params_object: Object with the neural net weights, biases
      and types
    init_dual_file: Path to file containing dual variables, if the path
      is empty, perform random initialization
      Expects numpy dictionary with
      lambda_pos_0, lambda_pos_1, ..
      lambda_neg_0, lambda_neg_1, ..
      lambda_quad_0, lambda_quad_1, ..
      lambda_lu_0, lambda_lu_1, ..
      random_init_variance: variance for random initialization
    init_nu: Value to initialize nu variable with

  Returns:
    dual_var: dual variables initialized appropriately.
  """
  lambda_pos = []
  lambda_neg = []
  lambda_quad = []
  lambda_lu = []

  if init_dual_file is None:
    for i in range(0, neural_net_params_object.num_hidden_layers + 1):
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_pos.append(tf.get_variable('lambda_pos_' + str(i),
                                        initializer=initializer,
                                        dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_neg.append(tf.get_variable('lambda_neg_' + str(i),
                                        initializer=initializer,
                                        dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_quad.append(tf.get_variable('lambda_quad_' + str(i),
                                         initializer=initializer,
                                         dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_lu.append(tf.get_variable('lambda_lu_' + str(i),
                                       initializer=initializer,
                                       dtype=tf.float32))
    nu = tf.get_variable('nu', initializer=init_nu)
    nu = tf.reshape(nu, shape=(1, 1))
  else:
    # Loading from file
    dual_var_init_val = np.load(init_dual_file).item()
    for i in range(0, neural_net_params_object.num_hidden_layers + 1):
      lambda_pos.append(
          tf.get_variable('lambda_pos_' + str(i),
                          initializer=dual_var_init_val['lambda_pos'][i],
                          dtype=tf.float32))
      lambda_neg.append(
          tf.get_variable('lambda_neg_' + str(i),
                          initializer=dual_var_init_val['lambda_neg'][i],
                          dtype=tf.float32))
      lambda_quad.append(
          tf.get_variable('lambda_quad_' + str(i),
                          initializer=dual_var_init_val['lambda_quad'][i],
                          dtype=tf.float32))
      lambda_lu.append(
          tf.get_variable('lambda_lu_' + str(i),
                          initializer=dual_var_init_val['lambda_lu'][i],
                          dtype=tf.float32))
    nu = tf.get_variable('nu', initializer=1.0*dual_var_init_val['nu'])
    nu = tf.reshape(nu, shape=(1, 1))
  dual_var = {'lambda_pos': lambda_pos, 'lambda_neg': lambda_neg,
              'lambda_quad': lambda_quad, 'lambda_lu': lambda_lu, 'nu': nu}
  return dual_var
