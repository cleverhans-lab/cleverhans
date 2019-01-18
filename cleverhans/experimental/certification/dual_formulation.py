"""Code with dual formulation for certification problem."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import numpy as np 
import tensorflow as tf
from numpy.linalg import cholesky 
from cleverhans.experimental.certification import utils

flags = tf.app.flags
FLAGS = flags.FLAGS

UPPER_BOUND = 1E5

class DualFormulation(object):
  """DualFormulation is a class that creates the dual objective function
  and access to matrix vector products for the matrix that is constrained
  to be Positive semidefinite """

  def __init__(self, sess, neural_net_param_object, test_input, true_class,
               adv_class, input_minval, input_maxval, epsilon):
    """Initializes dual formulation class.

    Args:
      sess: Tensorflow session
      neural_net_param_object: NeuralNetParam object created for
        the network under consideration
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
    self.final_linear = tf.reshape(self.final_linear,
                                   shape=[tf.size(self.final_linear), 1])
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
      current_lower = (0.5*(
          self.nn_params.forward_pass(self.lower[i] + self.upper[i], i)
          + self.nn_params.forward_pass(self.lower[i] - self.upper[i], i,
                                        is_abs=True))
                                 + self.nn_params.biases[i])
      current_upper = (0.5*(
          self.nn_params.forward_pass(self.lower[i] + self.upper[i], i)
          + self.nn_params.forward_pass(self.upper[i] -self.lower[i], i,
                                        is_abs=True))
                                 + self.nn_params.biases[i])
      self.pre_lower.append(current_lower)
      self.pre_upper.append(current_upper)
      self.lower.append(tf.nn.relu(current_lower))
      self.upper.append(tf.nn.relu(current_upper))

    # Using the preactivation lower and upper bounds 
    # to compute the linear regions 
    self.positive_indices = []
    self.negative_indices = []
    self.switch_indices = []

    for i in range(0, self.nn_params.num_hidden_layers + 1):
      # Positive index = 1 if the ReLU is always "on"
      self.positive_indices.append(tf.cast(self.pre_lower[i] >= 0, dtype=tf.float32))
      # Negative index = 1 if the ReLU is always off 
      self.negative_indices.append(tf.cast(self.pre_upper[i] <= 0, dtype=tf.float32))
      self.switch_indices.append(tf.cast(
        tf.multiply(self.pre_lower[i], self.pre_upper[i])<0, dtype=tf.float32))


    # Computing the optimization terms
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
      self.dual_index.append(self.dual_index[-1] +
                             self.nn_params.sizes[i])

  def initialize_dual(self, name, init_dual_folder=None, random_init_variance=0, init_nu=200.0):
    """Function to initialize the dual variables of the class 
      Args: 
        name: string for naming the dual variables created 
        init_dual_folder: Folder with numpy files for initializing the dual variables 
        random_init_variance: Variance for random initialization of the dual variables
        init_nu: initial valuye of variable nu
    """
    self.lambda_pos = []
    self.lambda_neg = []
    self.lambda_quad = []
    self.lambda_lu = []

    # Random initialization 
    if init_dual_folder is None:
      for i in range(self.nn_params.num_hidden_layers + 1):
        initializer = (np.random.uniform(0, random_init_variance, 
          size=(self.nn_params.sizes[i], 1))).astype(np.float32)
        self.lambda_pos.append(tf.get_variable(name + 'lambda_pos_' + str(i), 
          initializer=initializer, dtype=tf.float32))
        initializer = (np.random.uniform(0, random_init_variance, 
          size=(self.nn_params.sizes[i], 1))).astype(np.float32)
        self.lambda_neg.append(tf.get_variable(name + 'lambda_neg_' + str(i), 
          initializer=initializer, dtype=tf.float32))
        initializer = (np.random.uniform(0, random_init_variance, 
          size=(self.nn_params.sizes[i], 1))).astype(np.float32)
        self.lambda_quad.append(tf.get_variable(name + 'lambda_quad_' + str(i), 
          initializer=initializer, dtype=tf.float32))
        initializer = (np.random.uniform(0, random_init_variance, 
          size=(self.nn_params.sizes[i], 1))).astype(np.float32)
        self.lambda_lu.append(tf.get_variable(name + 'lambda_lu_' + str(i), 
          initializer=initializer, dtype=tf.float32))
      nu = tf.get_variable(name + 'nu', initializer=init_nu)
      self.nu = tf.reshape(nu, shape=(1, 1))
    # Loading from folder
    else:
      init_lambda_pos = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_pos.npy'))
      init_lambda_neg = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_neg.npy'))
      init_lambda_quad = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_quad.npy'))
      init_lambda_lu = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_lu.npy'))
      init_nu = np.load(os.path.join(FLAGS.init_dual_folder, 'nu.npy'))

      for i in range(0, self.nn_params.num_hidden_layers + 1):
        self.lambda_pos.append(
          tf.get_variable('lambda_pos_' + str(i),
                          initializer=init_lambda_pos[i],
                          dtype=tf.float32))
        self.lambda_neg.append(
          tf.get_variable('lambda_neg_' + str(i),
                          initializer=init_lambda_neg[i],
                          dtype=tf.float32))
        self.lambda_quad.append(
          tf.get_variable('lambda_quad_' + str(i),
                          initializer=init_lambda_quad[i],
                          dtype=tf.float32))
        self.lambda_lu.append(
          tf.get_variable('lambda_lu_' + str(i),
                          initializer=init_lambda_lu[i],
                          dtype=tf.float32))
      nu = tf.get_variable('nu', initializer=1.0*init_nu)
      self.nu = tf.reshape(nu, shape=(1, 1))
    self.dual_var = {'lambda_pos': self.lambda_pos, 'lambda_neg': self.lambda_neg, 
    'lambda_quad': self.lambda_quad, 'lambda_lu': self.lambda_lu, 'nu': self.nu}

  def initialize_placeholders(self):
    """Function to create placeholders for dual variables 
    to evaluate certificates """

    self.lambda_pos = []
    self.lambda_neg = []
    self.lambda_quad = []
    self.lambda_lu = []

    for i in range(self.nn_params.num_hidden_layers + 1):
      self.lambda_pos.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))
      self.lambda_neg.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))
      self.lambda_quad.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))
      self.lambda_lu.append(tf.placeholder(tf.float32, shape=(self.nn_params.sizes[i], 1)))
    self.nu = tf.placeholder(tf.float32, shape=(1, 1))


  def set_differentiable_objective(self):
    """Function that constructs minimization objective from dual variables."""
    # Checking if graphs are already created
    if self.vector_g is not None:
      return

    # Computing the scalar term
    bias_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers):
      bias_sum = bias_sum + tf.reduce_sum(
          tf.multiply(self.nn_params.biases[i], self.lambda_pos[i+1]))
    lu_sum = 0
    for i in range(0, self.nn_params.num_hidden_layers+1):
      lu_sum = lu_sum + tf.reduce_sum(
          tf.multiply(tf.multiply(self.lower[i], self.upper[i]),
                      self.lambda_lu[i]))

    self.scalar_f = - bias_sum - lu_sum + self.final_constant

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
    self.unconstrained_objective = self.scalar_f + 0.5*self.nu

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
    h_beta_rows = []
    for i in range(self.nn_params.num_hidden_layers):
      # Split beta of this block into [gamma, delta]
      gamma = beta[self.dual_index[i]:self.dual_index[i+1]]
      delta = beta[self.dual_index[i+1]:self.dual_index[i+2]]

      # Expanding the product with diagonal matrices
      if i == 0:
        h_beta_rows.append(tf.multiply(2*self.lambda_lu[i], gamma) -
                           self.nn_params.forward_pass(
                               tf.multiply(self.lambda_quad[i+1], delta),
                               i, is_transpose=True))
      else:
        h_beta_rows[i] = (h_beta_rows[i] +
                          tf.multiply(self.lambda_quad[i] +
                                      self.lambda_lu[i], gamma) -
                          self.nn_params.forward_pass(
                              tf.multiply(self.lambda_quad[i+1], delta),
                              i, is_transpose=True))

      new_row = (tf.multiply(self.lambda_quad[i+1] + self.lambda_lu[i+1], delta)
                 - tf.multiply(self.lambda_quad[i+1],
                               self.nn_params.forward_pass(gamma, i)))
      h_beta_rows.append(new_row)

    # Last boundary case
    h_beta_rows[self.nn_params.num_hidden_layers] = (
        h_beta_rows[self.nn_params.num_hidden_layers] +
        tf.multiply((self.lambda_quad[self.nn_params.num_hidden_layers] +
                     self.lambda_lu[self.nn_params.num_hidden_layers]),
                    delta))

    h_beta = tf.concat(h_beta_rows, axis=0)

    # Constructing final result using vector_g
    self.set_differentiable_objective()
    result = tf.concat([alpha*self.nu+tf.reduce_sum(
        tf.multiply(beta, self.vector_g))
                        , tf.multiply(alpha, self.vector_g) + h_beta], axis=0)
    return result

  
  def get_full_psd_matrix(self):
    """Function that retuns the tf graph corresponding to the entire matrix M.


    Returns:
      matrix_h: unrolled version of tf matrix corresponding to H
      matrix_m: unrolled tf matrix corresponding to M
    """
    # Computing the matrix term
    h_columns = []
    for i in range(self.nn_params.num_hidden_layers + 1):
      current_col_elems = []
      for j in range(i):
        current_col_elems.append(tf.zeros([self.nn_params.sizes[j],
                                           self.nn_params.sizes[i]]))

    # For the first layer, there is no relu constraint
      if i == 0:
        current_col_elems.append(utils.diag(self.lambda_lu[i]))
      else:
        current_col_elems.append(utils.diag(self.lambda_lu[i] +
                                            self.lambda_quad[i]))
      if i < self.nn_params.num_hidden_layers:
        current_col_elems.append((
            (tf.matmul(utils.diag(-1*self.lambda_quad[i+1]),
                       self.nn_params.weights[i]))))
      for j in range(i + 2, self.nn_params.num_hidden_layers + 1):
        current_col_elems.append(tf.zeros([self.nn_params.sizes[j],
                                           self.nn_params.sizes[i]]))
      current_column = tf.concat(current_col_elems, 0)
      h_columns.append(current_column)

    self.matrix_h = tf.concat(h_columns, 1)
    self.set_differentiable_objective()
    self.matrix_h = (self.matrix_h + tf.transpose(self.matrix_h))

    self.matrix_m = tf.concat([tf.concat([self.nu, tf.transpose(self.vector_g)],
                                         axis=1),
                               tf.concat([self.vector_g, self.matrix_h],
                                         axis=1)], axis=0)
    return self.matrix_h, self.matrix_m

  def save_dual(self, folder, sess):
    """Function to save the dual variables 
    Args:
      folder: The folder to save the dual variables 
      sess: current tensorflow session whose dual variables are to be saved 

    """ 
    if not tf.gfile.IsDirectory(folder):
      tf.gfile.MkDir(folder)
    [current_lambda_pos, current_lambda_neg, current_lambda_quad, 
    current_lambda_lu, current_nu] = sess.run([self.lambda_pos, 
      self.lambda_neg, self.lambda_quad, self.lambda_lu, self.nu])
    np.save(os.path.join(folder, 'lambda_pos'), current_lambda_pos)
    np.save(os.path.join(folder, 'lambda_neg'), current_lambda_neg)
    np.save(os.path.join(folder, 'lambda_lu'), current_lambda_lu)
    np.save(os.path.join(folder, 'lambda_quad'), current_lambda_quad)
    np.save(os.path.join(folder, 'nu'), current_nu)
    print('Saved the current dual variables in folder:', folder)

def compute_certificate(self, dual_folder):
  """ Function to compute the certificate based either current value
  or dual variables loaded from dual folder """
  if not dual_folder:
    lambda_pos_val = self.sess.run([x for x in self.lambda_pos])
    lambda_neg_val = self.sess.run([x for x in self.lambda_neg])
    lambda_quad_val = self.sess.run([x for x in self.lambda_quad])
    lambda_lu_val = self.sess.run([x for x in self.lambda_lu])
    nu_val = self.sess.run(self.nu)
  else:
    lambda_pos_val = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_pos.npy'))
    lambda_neg_val = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_neg.npy'))
    lambda_quad_val = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_quad.npy'))
    lambda_lu_val = np.load(os.path.join(FLAGS.init_dual_folder, 'lambda_lu.npy'))
    nu_val = np.load(os.path.join(FLAGS.init_dual_folder, 'nu.npy'))

  # Creating dictionary to feed placeholders 
  dual_feed_dict = {}
  dual_feed_dict.update({placeholder:value for placeholder, value in zip(self.lambda_pos, lambda_pos_val)})
  dual_feed_dict.update({placeholder:value for placeholder, value in zip(self.lambda_neg, lambda_neg_val)})
  dual_feed_dict.update({placeholder:value for placeholder, value in zip(self.lambda_quad, lambda_quad_val)})
  dual_feed_dict.update({placeholder:value for placeholder, value in zip(self.lambda_lu, lambda_lu_val)})
  dual_feed_dict.update({self.nu: nu_val})

  old_scalar_f = self.sess.run(self.scalar_f)
  old_matrix_h = self.sess.run(self.matrix_h)
  old_matrix_m = self.sess.run(self.matrix_m)

  min_eig_val_m, _ = eigs(matrix_m, k=1, which = 'LR', 
                                              tol=1E-5, sigma=-0.1)

  min_eig_val_m = min_eig_val_m - 1E-5

  dim = self.matrix_m_dimension
  try: 
    cholesky(old_matrix_m - np.real(min_eig_val_m)*np.eye(dim))
  except: 
    print("Increased min eigen value of M")
    min_eig_val_m = 2*min_eig_val_m
  else:
    pass

min_eig_val_h, _ = eigs(matrix_h, k=1, which = 'LR', 
                                              tol=1E-5, sigma=-0.1)

  min_eig_val_h = min_eig_val_h - 1E-5

  dim = self.matrix_m_dimension - 1
  try: 
    cholesky(old_matrix_h - np.real(min_eig_val_h)*np.eye(dim))
  except: 
    print("Increased min eigen value of H")
    min_eig_val_h = 2*min_eig_val_h
  else:
    pass

  current_certificate = UPPER_BOUND
  values = np.linspace(min_eig_val_m, min_eig_val_m, 5)
  for v in values:
    new_lambda_lu_val = [np.copy(x) for x in lambda_lu_val]
    for i in range(self.nn_params.num_hidden_layers + 1):
      new_lambda_lu_val = lambda_lu_val + 0.5*np.maximum(-values, 0) + 1E-6
    dual_feed_dict.update({placeholder:value for placeholder, value in zip(self.lambda_lu, new_lambda_lu_val)})
    scalar_f = self.sess.run(self.scalar_f, dual_feed_dict)
    vector_g = self.sess.run(self.vector_g, dual_feed_dict)
    matrix_h = self.sess.run(self.matrix_h, dual_feed_dict)
    second_term = np.matmul(np.matmul(np.transpose(new_vector_g), 
      inv(csc_matrix(new_matrix_h)).toarray()), new_vector_g) + 0.05
    dual_feed_dict.update({self.nu:second_term})
    check_psd_matrix_m = self.sess.run(matrix_m, dual_feed_dict)
    try: 
      cholesky(check_psd_matrix_m)
    except:
      print("Problem: Matrix is not PSD")
    else:
      print("All good: Matrix is PSD")
    computed_certificate = scalar_f + second_term 
    current_certificate = np.maximum(computed_certificate, current_certificate)

  return current_certificate


  



