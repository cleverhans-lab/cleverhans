"""Code with dual formulation for certification problem."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.sparse.linalg import eigs, LinearOperator
import tensorflow as tf
from tensorflow.contrib import autograph
import numpy as np

from cleverhans.experimental.certification import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# Tolerance value for eigenvalue computation
TOL = 1E-5

# Binary search constants
MAX_BINARY_SEARCH_ITER = 10
NU_UPDATE_CONSTANT = 1.3

# Bound on lowest value of certificate to check for numerical errors
LOWER_CERT_BOUND = -5.0
DEFAULT_LZS_PARAMS = {'min_iter': 5, 'max_iter': 50}


class DualFormulation(object):
  """DualFormulation is a class that creates the dual objective function
  and access to matrix vector products for the matrix that is constrained
  to be Positive semidefinite
  """

  def __init__(self, sess, dual_var, neural_net_param_object, test_input, true_class,
               adv_class, input_minval, input_maxval, epsilon,
               lzs_params=None, project_dual=True):
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
      lzs_params: Parameters for Lanczos algorithm (dictionary) in the form:
        {
          'min_iter': 5
          'max_iter': 50
        }
      project_dual: Whether we should create a projected dual object
    """
    self.sess = sess
    self.nn_params = neural_net_param_object
    self.test_input = tf.convert_to_tensor(test_input, dtype=tf.float32)
    self.true_class = true_class
    self.adv_class = adv_class
    self.input_minval = tf.convert_to_tensor(input_minval, dtype=tf.float32)
    self.input_maxval = tf.convert_to_tensor(input_maxval, dtype=tf.float32)
    self.epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    self.lzs_params = lzs_params or DEFAULT_LZS_PARAMS.copy()
    self.final_linear = (self.nn_params.final_weights[adv_class, :]
                         - self.nn_params.final_weights[true_class, :])
    self.final_linear = tf.reshape(
        self.final_linear, shape=[tf.size(self.final_linear), 1])
    self.final_constant = (self.nn_params.final_bias[adv_class]
                           - self.nn_params.final_bias[true_class])
    self.lanczos_dtype = tf.float64
    self.nn_dtype = tf.float32

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
      current_lower = 0.5 * (lo_plus_up + lo_minus_up) + self.nn_params.biases[i]
      current_upper = 0.5 * (lo_plus_up + up_minus_lo) + self.nn_params.biases[i]
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

    # Setup Lanczos functionality for compute certificate
    self.construct_lanczos_params()

    # Create projected dual object
    if project_dual:
      self.projected_dual = self.create_projected_dual()

  def create_projected_dual(self):
    """Function to create variables for the projected dual object.
    Function that projects the input dual variables onto the feasible set.
    Returns:
      projected_dual: Feasible dual solution corresponding to current dual
    """
    # TODO: consider whether we can use shallow copy of the lists without
    # using tf.identity
    projected_nu = tf.placeholder(tf.float32, shape=[])
    min_eig_h = tf.placeholder(tf.float32, shape=[])
    projected_lambda_pos = [tf.identity(x) for x in self.lambda_pos]
    projected_lambda_neg = [tf.identity(x) for x in self.lambda_neg]
    projected_lambda_quad = [
        tf.identity(x) for x in self.lambda_quad
    ]
    projected_lambda_lu = [tf.identity(x) for x in self.lambda_lu]

    for i in range(self.nn_params.num_hidden_layers + 1):
      # Making H PSD
      projected_lambda_lu[i] = self.lambda_lu[i] + 0.5*tf.maximum(-min_eig_h, 0) + TOL
      # Adjusting the value of \lambda_neg to make change in g small
      projected_lambda_neg[i] = self.lambda_neg[i] + tf.multiply(
          (self.lower[i] + self.upper[i]),
          (self.lambda_lu[i] - projected_lambda_lu[i]))
      projected_lambda_neg[i] = (tf.multiply(self.negative_indices[i],
                                             projected_lambda_neg[i]) +
                                 tf.multiply(self.switch_indices[i],
                                             tf.maximum(projected_lambda_neg[i], 0)))

    projected_dual_var = {
        'lambda_pos': projected_lambda_pos,
        'lambda_neg': projected_lambda_neg,
        'lambda_lu': projected_lambda_lu,
        'lambda_quad': projected_lambda_quad,
        'nu': projected_nu,
    }
    projected_dual_object = DualFormulation(
        self.sess, projected_dual_var, self.nn_params,
        self.test_input, self.true_class,
        self.adv_class, self.input_minval,
        self.input_maxval, self.epsilon,
        self.lzs_params,
        project_dual=False)
    projected_dual_object.min_eig_val_h = min_eig_h
    return projected_dual_object

  def construct_lanczos_params(self):
    """Computes matrices T and V using the Lanczos algorithm.

    Args:
      k: number of iterations and dimensionality of the tridiagonal matrix
    Returns:
      eig_vec: eigen vector corresponding to min eigenvalue
    """
    # Using autograph to automatically handle
    # the control flow of minimum_eigen_vector
    self.min_eigen_vec = autograph.to_graph(utils.tf_lanczos_smallest_eigval)

    def _m_vector_prod_fn(x):
      return self.get_psd_product(x, dtype=self.lanczos_dtype)
    def _h_vector_prod_fn(x):
      return self.get_h_product(x, dtype=self.lanczos_dtype)

    # Construct nodes for computing eigenvalue of M
    self.m_min_vec_estimate = np.zeros(shape=(self.matrix_m_dimension, 1), dtype=np.float64)
    zeros_m = tf.zeros(shape=(self.matrix_m_dimension, 1), dtype=tf.float64)
    self.m_min_vec_ph = tf.placeholder_with_default(input=zeros_m,
                                                    shape=(self.matrix_m_dimension, 1),
                                                    name='m_min_vec_ph')
    self.m_min_eig, self.m_min_vec = self.min_eigen_vec(_m_vector_prod_fn,
                                                        self.matrix_m_dimension,
                                                        self.m_min_vec_ph,
                                                        self.lzs_params['max_iter'],
                                                        dtype=self.lanczos_dtype)
    self.m_min_eig = tf.cast(self.m_min_eig, self.nn_dtype)
    self.m_min_vec = tf.cast(self.m_min_vec, self.nn_dtype)

    self.h_min_vec_estimate = np.zeros(shape=(self.matrix_m_dimension - 1, 1), dtype=np.float64)
    zeros_h = tf.zeros(shape=(self.matrix_m_dimension - 1, 1), dtype=tf.float64)
    self.h_min_vec_ph = tf.placeholder_with_default(input=zeros_h,
                                                    shape=(self.matrix_m_dimension - 1, 1),
                                                    name='h_min_vec_ph')
    self.h_min_eig, self.h_min_vec = self.min_eigen_vec(_h_vector_prod_fn,
                                                        self.matrix_m_dimension-1,
                                                        self.h_min_vec_ph,
                                                        self.lzs_params['max_iter'],
                                                        dtype=self.lanczos_dtype)
    self.h_min_eig = tf.cast(self.h_min_eig, self.nn_dtype)
    self.h_min_vec = tf.cast(self.h_min_vec, self.nn_dtype)

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

  def get_h_product(self, vector, dtype=None):
    """Function that provides matrix product interface with PSD matrix.

    Args:
      vector: the vector to be multiplied with matrix H

    Returns:
      result_product: Matrix product of H and vector
    """
    # Computing the product of matrix_h with beta (input vector)
    # At first layer, h is simply diagonal
    if dtype is None:
      dtype = self.nn_dtype
    beta = tf.cast(vector, self.nn_dtype)
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
    return tf.cast(h_beta, dtype)

  def get_psd_product(self, vector, dtype=None):
    """Function that provides matrix product interface with PSD matrix.

    Args:
      vector: the vector to be multiplied with matrix M

    Returns:
      result_product: Matrix product of M and vector
    """
    # For convenience, think of x as [\alpha, \beta]
    if dtype is None:
      dtype = self.nn_dtype
    vector = tf.cast(vector, self.nn_dtype)
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
    return tf.cast(result, dtype)

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
            tf.concat([tf.reshape(self.nu, (1, 1)), tf.transpose(self.vector_g)], axis=1),
            tf.concat([self.vector_g, self.matrix_h], axis=1)
        ],
        axis=0)
    return self.matrix_h, self.matrix_m

  def make_m_psd(self, original_nu, feed_dictionary):
    """Run binary search to find a value for nu that makes M PSD
    Args:
      original_nu: starting value of nu to do binary search on
      feed_dictionary: dictionary of updated lambda variables to feed into M
    Returns:
      new_nu: new value of nu
    """
    feed_dict = feed_dictionary.copy()
    _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)

    lower_nu = original_nu
    upper_nu = original_nu
    num_iter = 0

    # Find an upper bound on nu
    while min_eig_val_m - TOL < 0 and num_iter < (MAX_BINARY_SEARCH_ITER / 2):
      num_iter += 1
      upper_nu *= NU_UPDATE_CONSTANT
      feed_dict.update({self.nu: upper_nu})
      _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)

    final_nu = upper_nu

    # Perform binary search to find best value of nu
    while lower_nu <= upper_nu and num_iter < MAX_BINARY_SEARCH_ITER:
      num_iter += 1
      mid_nu = (lower_nu + upper_nu) / 2
      feed_dict.update({self.nu: mid_nu})
      _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)
      if min_eig_val_m - TOL < 0:
        lower_nu = mid_nu
      else:
        upper_nu = mid_nu

    final_nu = upper_nu

    return final_nu

  def get_lanczos_eig(self, compute_m=True, feed_dict=None):
    """Computes the min eigen value and corresponding vector of matrix M or H
    using the Lanczos algorithm.
    Args:
      compute_m: boolean to determine whether we should compute eig val/vec
        for M or for H. True for M; False for H.
      feed_dict: dictionary mapping from TF placeholders to values (optional)
    Returns:
      min_eig_vec: Corresponding eigen vector to min eig val
      eig_val: Minimum eigen value
    """
    if compute_m:
      min_eig, min_vec = self.sess.run([self.m_min_eig, self.m_min_vec], feed_dict=feed_dict)

    else:
      min_eig, min_vec = self.sess.run([self.h_min_eig, self.h_min_vec], feed_dict=feed_dict)

    return min_vec, min_eig

  def compute_certificate(self, current_step, feed_dictionary):
    """ Function to compute the certificate based either current value
    or dual variables loaded from dual folder """
    feed_dict = feed_dictionary.copy()
    nu = feed_dict[self.nu]
    second_term = self.make_m_psd(nu, feed_dict)
    tf.logging.info('Nu after modifying: ' + str(second_term))
    feed_dict.update({self.nu: second_term})
    computed_certificate = self.sess.run(self.unconstrained_objective, feed_dict=feed_dict)

    tf.logging.info('Inner step: %d, current value of certificate: %f',
                    current_step, computed_certificate)

    # Sometimes due to either overflow or instability in inverses,
    # the returned certificate is large and negative -- keeping a check
    if LOWER_CERT_BOUND < computed_certificate < 0:
      _, min_eig_val_m = self.get_lanczos_eig(feed_dict=feed_dict)
      tf.logging.info('min eig val from lanczos: ' + str(min_eig_val_m))
      input_vector_m = tf.placeholder(tf.float32, shape=(self.matrix_m_dimension, 1))
      output_vector_m = self.get_psd_product(input_vector_m)

      def np_vector_prod_fn_m(np_vector):
        np_vector = np.reshape(np_vector, [-1, 1])
        feed_dict.update({input_vector_m:np_vector})
        output_np_vector = self.sess.run(output_vector_m, feed_dict=feed_dict)
        return output_np_vector
      linear_operator_m = LinearOperator((self.matrix_m_dimension,
                                          self.matrix_m_dimension),
                                         matvec=np_vector_prod_fn_m)
      # Performing shift invert scipy operation when eig val estimate is available
      min_eig_val_m_scipy, _ = eigs(linear_operator_m, k=1, which='SR', tol=TOL)

      tf.logging.info('min eig val m from scipy: ' + str(min_eig_val_m_scipy))

      if min_eig_val_m - TOL > 0:
        tf.logging.info('Found certificate of robustness!')
        return True

    return False
