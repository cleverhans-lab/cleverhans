"""Code with dual formulation for certification problem."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from cleverhans.experimental.certification import utils


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
    self.pre_lower.append(self.lower[0])
    self.pre_upper.append(self.upper[0])
    # Initializing at the input layer with \ell_\infty constraints
    self.lower.append(
        tf.maximum(self.test_input - self.epsilon, self.input_minval))
    self.upper.append(
        tf.minimum(self.test_input + self.epsilon, self.input_maxval))
    for i in range(0, self.nn_params.num_hidden_layers):
      current_lower = 0.5*(
          self.nn_params.forward_pass(self.lower[i] + self.upper[i], i)
          + self.nn_params.forward_pass(self.lower[i] - self.upper[i], i,
                                        is_abs=True))
                                 + self.nn_params.biases[i]
      current_upper = 0.5*(
          self.nn_params.forward_pass(self.lower[i] + self.upper[i], i)
          + self.nn_params.forward_pass(self.upper[i] -self.lower[i], i,
                                        is_abs=True))
                                 + self.nn_params.biases[i]
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
    if init_folder is None:
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
    self.dual_var{'lambda_pos': self.lambda_pos, 'lambda_neg': self.lambda_neg, 
    'lambda_quad': self.lambda_quad, 'lambda_lu': self.lambda_lu, 'nu': self.nu}

  def initialize_placeholder(self):
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

  def compute_certificate(self):
    """Function to compute the certificate associated with feasible solution."""
    self.set_differentiable_objective()
    self.get_full_psd_matrix()
    # TODO: replace matrix_inverse with functin which uses matrix-vector product
    projected_certificate = (
        self.scalar_f +
        0.5*tf.matmul(tf.matmul(tf.transpose(self.vector_g),
                                tf.matrix_inverse(self.matrix_h)),
                      self.vector_g))
    return projected_certificate

  def project_dual(self):
    """Function that projects the input dual variables onto the feasible set.

    Returns:
      projected_dual: Feasible dual solution corresponding to current dual
      projected_certificate: Objective value of feasible dual
    """
    # TODO: consider whether we can use shallow copy of the lists without
    # using tf.identity
    projected_lambda_pos = [tf.identity(x) for x in self.lambda_pos]
    projected_lambda_neg = [tf.identity(x) for x in self.lambda_neg]
    projected_lambda_quad = [tf.identity(x) for x in self.lambda_quad]
    projected_lambda_lu = [tf.identity(x) for x in self.lambda_lu]
    projected_nu = tf.identity(self.nu)

    # TODO: get rid of the special case for one hidden layer
    # Different projection for 1 hidden layer
    if self.nn_params.num_hidden_layers == 1:
      # Creating equivalent PSD matrix for H by Schur complements
      diag_entries = 0.5*tf.divide(
          tf.square(self.lambda_quad[self.nn_params.num_hidden_layers]),
          (self.lambda_quad[self.nn_params.num_hidden_layers] +
           self.lambda_lu[self.nn_params.num_hidden_layers]))
      # If lambda_quad[i], lambda_lu[i] are 0, entry is NaN currently,
      # but we want to set that to 0
      diag_entries = tf.where(tf.is_nan(diag_entries),
                              tf.zeros_like(diag_entries), diag_entries)
      matrix = (
          tf.matmul(tf.matmul(tf.transpose(
              self.nn_params.weights[self.nn_params.num_hidden_layers-1]),
                              utils.diag(diag_entries)),
                    self.nn_params.weights[self.nn_params.num_hidden_layers-1]))
      new_matrix = utils.diag(
          2*self.lambda_lu[self.nn_params.num_hidden_layers - 1]) - matrix
      # Making symmetric
      new_matrix = 0.5*(new_matrix + tf.transpose(new_matrix))
      eig_vals = tf.self_adjoint_eigvals(new_matrix)
      min_eig = tf.reduce_min(eig_vals)
      # If min_eig is positive, already feasible, so don't add
      # Otherwise add to make PSD [1E-6 is for ensuring strictly PSD (useful
      # while inverting)
      projected_lambda_lu[0] = (projected_lambda_lu[0] +
                                0.5*tf.maximum(-min_eig, 0) + 1E-6)

    else:
      # Minimum eigen value of H
      # TODO: Write this in terms of matrix multiply
      # matrix H is a submatrix of M, thus we just need to extend existing code
      # for computing matrix-vector product (see get_psd_product function).
      # Then use the same trick to compute smallest eigenvalue.
      eig_vals = tf.self_adjoint_eigvals(self.matrix_h)
      min_eig = tf.reduce_min(eig_vals)

      for i in range(self.nn_params.num_hidden_layers+1):
        # Since lambda_lu appears only in diagonal terms, can subtract to
        # make PSD and feasible
        projected_lambda_lu[i] = (projected_lambda_lu[i] +
                                  0.5*tf.maximum(-min_eig, 0) + 1E-6)
        # Adjusting lambda_neg wherever possible so that lambda_neg + lambda_lu
        # remains close to unchanged
        # projected_lambda_neg[i] = tf.maximum(0.0, projected_lambda_neg[i] +
        #                                     (0.5*min_eig - 1E-6)*
        #                                     (self.lower[i] + self.upper[i]))

    projected_dual_var = {'lambda_pos': projected_lambda_pos,
                          'lambda_neg': projected_lambda_neg,
                          'lambda_lu': projected_lambda_lu,
                          'lambda_quad': projected_lambda_quad,
                          'nu': projected_nu}
    projected_dual_object = DualFormulation(projected_dual_var,
                                            self.nn_params,
                                            self.test_input,
                                            self.true_class,
                                            self.adv_class,
                                            self.input_minval,
                                            self.input_maxval,
                                            self.epsilon)
    projected_certificate = projected_dual_object.compute_certificate()
    return projected_certificate

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
