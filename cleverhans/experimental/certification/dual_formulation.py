"""Code with dual formulation for certification problem."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg import lgmres

from cleverhans.experimental.certification import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# Tolerance value for eigenvalue computation
TOL = 1E-5


class DualFormulation(object):
  """DualFormulation is a class that creates the dual objective function
  and access to matrix vector products for the matrix that is constrained
  to be Positive semidefinite
  """

  def __init__(self, sess, dual_var, neural_net_param_object, test_input, true_class,
               adv_class, input_minval, input_maxval, epsilon):
    """Initializes dual formulation class.

    Args:
      sess: Tensorflow session
      dual_var: dictionary of dual variables containing a) lambda_pos
        b) lambda_neg, c) lambda_quad, d) lambda_lu
      neural_net_param_object: NeuralNetParam object created for the network
        under consideration
      test_input: clean example to certify around
      true_class: the class label of the test input
      adv_class: the label that the adversary tried to perturb input to
      input_minval: minimum value of valid input range
      input_maxval: maximum value of valid input range
      epsilon: Size of the perturbation (scaled for [0, 1] input)
    """
    self.sess = sess
    self.nn_params = neural_net_param_object
    self.test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
    self.true_class = true_class
    self.adv_class = adv_class
    self.input_minval = tf.convert_to_tensor(input_minval, dtype=tf.float32)
    self.input_maxval = tf.convert_to_tensor(input_maxval, dtype=tf.float32)
    self.epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    self.final_linear = (self.nn_params.final_weights[adv_class, :]
                         - self.nn_params.final_weights[true_class, :])
    self.final_linear = tf.reshape(
        self.final_linear, shape=[tf.size(self.final_linear), 1])
    self.final_constant = (self.nn_params.final_bias[adv_class]
                           - self.nn_params.final_bias[true_class])

    # Computing lower and upper bounds
    # Note that lower and upper are of size nn_params.num_hidden_layers + 1
    self.lower = []
    self.upper = []

    # Also computing pre activation lower and upper bounds
    # to compute always-off and always-on units
    self.pre_lower = []
    self.pre_upper = []

    # Initializing at the input layer with \ell_\infty constraints
    self.lower.append(
        tf.maximum(self.test_input - self.epsilon, self.input_minval))
    self.upper.append(
        tf.minimum(self.test_input + self.epsilon, self.input_maxval))
    self.pre_lower.append(self.lower[0])
    self.pre_upper.append(self.upper[0])

    for i in range(0, self.nn_params.num_hidden_layers):
      lo_plus_up = self.nn_params.forward_pass(self.lower[i] + self.upper[i], i)
      lo_minus_up = self.nn_params.forward_pass(self.lower[i] - self.upper[i], i, is_abs=True)
      up_minus_lo = self.nn_params.forward_pass(self.upper[i] - self.lower[i], i, is_abs=True)
      current_lower = 0.5 * (lo_plus_up + lo_minus_up + self.nn_params.biases[i])
      current_upper = 0.5 * (lo_plus_up + up_minus_lo + self.nn_params.biases[i])
      self.pre_lower.append(current_lower)
      self.pre_upper.append(current_upper)
      self.lower.append(tf.nn.relu(current_lower))
      self.upper.append(tf.nn.relu(current_upper))

    # Run lower and upper because they don't change
    self.pre_lower = self.sess.run(self.pre_lower)
    self.pre_upper = self.sess.run(self.pre_upper)
    self.lower = self.sess.run(self.lower)
    self.upper = self.sess.run(self.upper)

    # Using the preactivation lower and upper bounds
    # to compute the linear regions
    self.positive_indices = []
    self.negative_indices = []
    self.switch_indices = []

    for i in range(0, self.nn_params.num_hidden_layers + 1):
      # Positive index = 1 if the ReLU is always "on"
      self.positive_indices.append(np.asarray(self.pre_lower[i] >= 0, dtype=np.float32))
      # Negative index = 1 if the ReLU is always off
      self.negative_indices.append(np.asarray(self.pre_upper[i] <= 0, dtype=np.float32))
      # Switch index = 1 if the ReLU could be either on or off
      self.switch_indices.append(np.asarray(
          np.multiply(self.pre_lower[i], self.pre_upper[i]) < 0, dtype=np.float32))

    # Computing the optimization terms
    self.lambda_pos = [x for x in dual_var['lambda_pos']]
    self.lambda_neg = [x for x in dual_var['lambda_neg']]
    self.lambda_quad = [x for x in dual_var['lambda_quad']]
    self.lambda_lu = [x for x in dual_var['lambda_lu']]
    self.nu = dual_var['nu']
    self.vector_g = None
    self.scalar_f = None
    self.matrix_h = None
    self.matrix_m = None
    self.matrix_m_dimension = 1 + np.sum(self.nn_params.sizes)

    # The primal vector in the SDP can be thought of as [layer_1, layer_2..]
    # In this concatenated version, dual_index[i] that marks the start
    # of layer_i
    # This is useful while computing implicit products with matrix H
    self.dual_index = [0]
    for i in range(self.nn_params.num_hidden_layers + 1):
      self.dual_index.append(self.dual_index[-1] + self.nn_params.sizes[i])

    # Construct objectives, matrices, and certificate
    self.set_differentiable_objective()
    if not self.nn_params.has_conv:
      self.get_full_psd_matrix()

  def set_differentiable_objective(self):
    """Function that constructs minimization objective from dual variables."""
    # Checking if graphs are already created
    if self.vector_g is not None:
      return

    # Computing the scalar term
    bias_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers):
      bias_sum = bias_sum + tf.reduce_sum(
          tf.multiply(self.nn_params.biases[i], self.lambda_pos[i + 1]))
    lu_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers + 1):
      lu_sum = lu_sum + tf.reduce_sum(
          tf.multiply(tf.multiply(self.lower[i], self.upper[i]),
                      self.lambda_lu[i]))

    self.scalar_f = -bias_sum - lu_sum + self.final_constant

    # Computing the vector term
    g_rows = []
    for i in range(0, self.nn_params.num_hidden_layers):
      if i > 0:
        current_row = (self.lambda_neg[i] + self.lambda_pos[i] -
                       self.nn_params.forward_pass(self.lambda_pos[i+1],
                                                   i, is_transpose=True) +
                       tf.multiply(self.lower[i]+self.upper[i],
                                   self.lambda_lu[i]) +
                       tf.multiply(self.lambda_quad[i],
                                   self.nn_params.biases[i-1]))
      else:
        current_row = (-self.nn_params.forward_pass(self.lambda_pos[i+1],
                                                    i, is_transpose=True)
                       + tf.multiply(self.lower[i]+self.upper[i],
                                     self.lambda_lu[i]))
      g_rows.append(current_row)

    # Term for final linear term
    g_rows.append((self.lambda_pos[self.nn_params.num_hidden_layers] +
                   self.lambda_neg[self.nn_params.num_hidden_layers] +
                   self.final_linear +
                   tf.multiply((self.lower[self.nn_params.num_hidden_layers]+
                                self.upper[self.nn_params.num_hidden_layers]),
                               self.lambda_lu[self.nn_params.num_hidden_layers])
                   + tf.multiply(
                       self.lambda_quad[self.nn_params.num_hidden_layers],
                       self.nn_params.biases[
                           self.nn_params.num_hidden_layers-1])))
    self.vector_g = tf.concat(g_rows, axis=0)
    self.unconstrained_objective = self.scalar_f + 0.5 * self.nu

  def get_h_product(self, vector):
    """Function that provides matrix product interface with PSD matrix.

    Args:
      vector: the vector to be multiplied with matrix H

    Returns:
      result_product: Matrix product of H and vector
    """
    # Computing the product of matrix_h with beta (input vector)
    # At first layer, h is simply diagonal
    beta = vector
    h_beta_rows = []
    for i in range(self.nn_params.num_hidden_layers):
      # Split beta of this block into [gamma, delta]
      gamma = beta[self.dual_index[i]:self.dual_index[i + 1]]
      delta = beta[self.dual_index[i + 1]:self.dual_index[i + 2]]

      # Expanding the product with diagonal matrices
      if i == 0:
        h_beta_rows.append(
            tf.multiply(2 * self.lambda_lu[i], gamma) -
            self.nn_params.forward_pass(
                tf.multiply(self.lambda_quad[i + 1], delta),
                i,
                is_transpose=True))
      else:
        h_beta_rows[i] = (h_beta_rows[i] +
                          tf.multiply(self.lambda_quad[i] +
                                      self.lambda_lu[i], gamma) -
                          self.nn_params.forward_pass(
                              tf.multiply(self.lambda_quad[i+1], delta),
                              i, is_transpose=True))

      new_row = (
          tf.multiply(self.lambda_quad[i + 1] + self.lambda_lu[i + 1], delta) -
          tf.multiply(self.lambda_quad[i + 1],
                      self.nn_params.forward_pass(gamma, i)))
      h_beta_rows.append(new_row)

    # Last boundary case
    h_beta_rows[self.nn_params.num_hidden_layers] = (
        h_beta_rows[self.nn_params.num_hidden_layers] +
        tf.multiply((self.lambda_quad[self.nn_params.num_hidden_layers] +
                     self.lambda_lu[self.nn_params.num_hidden_layers]),
                    delta))

    h_beta = tf.concat(h_beta_rows, axis=0)
    return h_beta

  def get_psd_product(self, vector):
    """Function that provides matrix product interface with PSD matrix.

    Args:
      vector: the vector to be multiplied with matrix M

    Returns:
      result_product: Matrix product of M and vector
    """
    # For convenience, think of x as [\alpha, \beta]
    alpha = tf.reshape(vector[0], shape=[1, 1])
    beta = vector[1:]
    # Computing the product of matrix_h with beta part of vector
    # At first layer, h is simply diagonal
    h_beta = self.get_h_product(beta)

    # Constructing final result using vector_g
    result = tf.concat(
        [
            alpha * self.nu + tf.reduce_sum(tf.multiply(beta, self.vector_g)),
            tf.multiply(alpha, self.vector_g) + h_beta
        ],
        axis=0)
    return result

  def get_full_psd_matrix(self):
    """Function that returns the tf graph corresponding to the entire matrix M.

    Returns:
      matrix_h: unrolled version of tf matrix corresponding to H
      matrix_m: unrolled tf matrix corresponding to M
    """
    if self.matrix_m is not None:
      return self.matrix_h, self.matrix_m

    # Computing the matrix term
    h_columns = []
    for i in range(self.nn_params.num_hidden_layers + 1):
      current_col_elems = []
      for j in range(i):
        current_col_elems.append(
            tf.zeros([self.nn_params.sizes[j], self.nn_params.sizes[i]]))

    # For the first layer, there is no relu constraint
      if i == 0:
        current_col_elems.append(utils.diag(self.lambda_lu[i]))
      else:
        current_col_elems.append(
            utils.diag(self.lambda_lu[i] + self.lambda_quad[i]))
      if i < self.nn_params.num_hidden_layers:
        current_col_elems.append(tf.matmul(
            utils.diag(-1 * self.lambda_quad[i + 1]),
            self.nn_params.weights[i]))
      for j in range(i + 2, self.nn_params.num_hidden_layers + 1):
        current_col_elems.append(
            tf.zeros([self.nn_params.sizes[j], self.nn_params.sizes[i]]))
      current_column = tf.concat(current_col_elems, 0)
      h_columns.append(current_column)

    self.matrix_h = tf.concat(h_columns, 1)
    self.matrix_h = (self.matrix_h + tf.transpose(self.matrix_h))

    self.matrix_m = tf.concat(
        [
            tf.concat([self.nu, tf.transpose(self.vector_g)], axis=1),
            tf.concat([self.vector_g, self.matrix_h], axis=1)
        ],
        axis=0)
    return self.matrix_h, self.matrix_m

  def compute_certificate(self):
    """ Function to compute the certificate based either current value
    or dual variables loaded from dual folder """
    lambda_neg_val = self.sess.run(self.lambda_neg)
    lambda_lu_val = self.sess.run(self.lambda_lu)

    input_vector_h = tf.placeholder(tf.float32, shape=(self.matrix_m_dimension - 1, 1))
    output_vector_h = self.get_h_product(input_vector_h)

    def np_vector_prod_fn_h(np_vector):
      np_vector = np.reshape(np_vector, [-1, 1])
      output_np_vector = self.sess.run(output_vector_h, feed_dict={input_vector_h:np_vector})
      return output_np_vector
    linear_operator_h = LinearOperator((self.matrix_m_dimension - 1,
                                        self.matrix_m_dimension - 1),
                                       matvec=np_vector_prod_fn_h)
    # Performing shift invert scipy operation when eig val estimate is available
    min_eig_val_h, _ = eigs(linear_operator_h,
                            k=1, which='SR', tol=TOL)

    # It's likely that the approximation is off by the tolerance value,
    # so we shift it back
    min_eig_val_h = np.real(min_eig_val_h) - TOL

    dual_feed_dict = {}

    new_lambda_lu_val = [np.copy(x) for x in lambda_lu_val]
    new_lambda_neg_val = [np.copy(x) for x in lambda_neg_val]

    for i in range(self.nn_params.num_hidden_layers + 1):
      # Making H PSD
      new_lambda_lu_val[i] = lambda_lu_val[i] + 0.5*np.maximum(-min_eig_val_h, 0) + TOL
      # Adjusting the value of \lambda_neg to make change in g small
      new_lambda_neg_val[i] = lambda_neg_val[i] + np.multiply((self.lower[i] + self.upper[i]),
                                                              (lambda_lu_val[i] -
                                                               new_lambda_lu_val[i]))
      new_lambda_neg_val[i] = (np.multiply(self.negative_indices[i],
                                           new_lambda_neg_val[i]) +
                               np.multiply(self.switch_indices[i],
                                           np.maximum(new_lambda_neg_val[i], 0)))

    dual_feed_dict.update(zip(self.lambda_lu, new_lambda_lu_val))
    dual_feed_dict.update(zip(self.lambda_neg, new_lambda_neg_val))
    scalar_f = self.sess.run(self.scalar_f, feed_dict=dual_feed_dict)
    vector_g = self.sess.run(self.vector_g, feed_dict=dual_feed_dict)
    x, _ = lgmres(linear_operator_h, vector_g)
    x = x.reshape((x.shape[0], 1))
    second_term = np.matmul(np.transpose(vector_g), x) + 0.05

    computed_certificate = scalar_f + 0.5*second_term

    return computed_certificate
