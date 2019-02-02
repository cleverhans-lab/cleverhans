"""Code for setting up the optimization problem for certification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import autograph
from cleverhans.experimental.certification import utils
from cleverhans.experimental.certification import dual_formulation

# Bound on lowest value of certificate to check for numerical errors
LOWER_CERT_BOUND = -10.0


class Optimization(object):
  """Class that sets up and runs the optimization of dual_formulation"""

  def __init__(self, dual_formulation_object, sess, optimization_params):
    """Initialize the class variables.

    Args:
      dual_formulation_object: Instance of DualFormulation that contains the
        dual variables and objective
      sess: tf session to be used to run
      optimization_params: Dictionary with the following
        eig_num_iter - Number of iterations to run for computing minimum eigen
          value
        eig_learning_rate - Learning rate for minimum eigen value iterations
        init_smooth - Starting value of the smoothness parameter (typically
        around 0.001)
        smooth_decay - The factor by which to decay after every outer loop epoch
        optimizer - one of gd, adam, momentum or adagrad
    """
    self.dual_object = dual_formulation_object
    self.projected_dual_object = self.project_dual()
    self.params = optimization_params
    self.penalty_placeholder = tf.placeholder(tf.float32, shape=[])

    # The dimensionality of matrix M is the sum of sizes of all layers + 1
    # The + 1 comes due to a row and column of M representing the linear terms
    self.eig_init_vec_placeholder = tf.placeholder(
        tf.float32, shape=[1 + self.dual_object.dual_index[-1], 1])
    self.smooth_placeholder = tf.placeholder(tf.float32, shape=[])
    self.eig_num_iter_placeholder = tf.placeholder(tf.int32, shape=[])
    self.sess = sess

    # Create graph for optimization
    self.prepare_for_optimization()

  def project_dual(self):
    """Function to create variables for the projected dual object.
    Function that projects the input dual variables onto the feasible set.

    Returns:
      projected_dual: Feasible dual solution corresponding to current dual
    """
    # TODO: consider whether we can use shallow copy of the lists without
    # using tf.identity
    projected_lambda_pos = [tf.identity(x) for x in self.dual_object.lambda_pos]
    projected_lambda_neg = [tf.identity(x) for x in self.dual_object.lambda_neg]
    projected_lambda_quad = [
        tf.identity(x) for x in self.dual_object.lambda_quad
    ]
    projected_lambda_lu = [tf.identity(x) for x in self.dual_object.lambda_lu]
    projected_nu = tf.identity(self.dual_object.nu)

    # TODO: get rid of the special case for one hidden layer
    # Different projection for 1 hidden layer
    if self.dual_object.nn_params.num_hidden_layers == 1:
      # Creating equivalent PSD matrix for H by Schur complements
      diag_entries = 0.5*tf.divide(
          tf.square(self.dual_object.lambda_quad[self.dual_object.nn_params.num_hidden_layers]),
          (self.dual_object.lambda_quad[self.dual_object.nn_params.num_hidden_layers] +
           self.dual_object.lambda_lu[self.dual_object.nn_params.num_hidden_layers]))
      # If lambda_quad[i], lambda_lu[i] are 0, entry is NaN currently,
      # but we want to set that to 0
      diag_entries = tf.where(
          tf.is_nan(diag_entries), tf.zeros_like(diag_entries), diag_entries)
      last_layer_weights = self.dual_object.nn_params.weights[
          self.dual_object.nn_params.num_hidden_layers - 1]
      matrix = tf.matmul(
          tf.matmul(tf.transpose(last_layer_weights),
                    utils.diag(diag_entries)),
          last_layer_weights)
      new_matrix = utils.diag(2 * self.dual_object.lambda_lu[
          self.dual_object.nn_params.num_hidden_layers - 1]) - matrix
      # Making symmetric
      new_matrix = 0.5 * (new_matrix + tf.transpose(new_matrix))
      eig_vals = tf.self_adjoint_eigvals(new_matrix)
      min_eig = tf.reduce_min(eig_vals)
      # If min_eig is positive, already feasible, so don't add
      # Otherwise add to make PSD [1E-6 is for ensuring strictly PSD (useful
      # while inverting)
      projected_lambda_lu[0] = (
          projected_lambda_lu[0] + 0.5 * tf.maximum(-min_eig, 0) + 1E-6)

    else:
      # Minimum eigen value of H
      # TODO: Write this in terms of matrix multiply
      # matrix H is a submatrix of M, thus we just need to extend existing code
      # for computing matrix-vector product (see get_psd_product function).
      # Then use the same trick to compute smallest eigenvalue.
      eig_vals = tf.self_adjoint_eigvals(self.dual_object.matrix_h)
      min_eig = tf.reduce_min(eig_vals)

      for i in range(self.dual_object.nn_params.num_hidden_layers + 1):
        # Since lambda_lu appears only in diagonal terms, can subtract to
        # make PSD and feasible
        projected_lambda_lu[i] = (
            projected_lambda_lu[i] + 0.5 * tf.maximum(-min_eig, 0) + 1E-6)
        # Adjusting lambda_neg wherever possible so that lambda_neg + lambda_lu
        # remains close to unchanged
        # projected_lambda_neg[i] = tf.maximum(0.0, projected_lambda_neg[i] +
        #                                     (0.5*min_eig - 1E-6)*
        #                                     (self.lower[i] + self.upper[i]))

    projected_dual_var = {
        'lambda_pos': projected_lambda_pos,
        'lambda_neg': projected_lambda_neg,
        'lambda_lu': projected_lambda_lu,
        'lambda_quad': projected_lambda_quad,
        'nu': projected_nu
    }
    projected_dual_object = dual_formulation.DualFormulation(
        projected_dual_var, self.dual_object.nn_params,
        self.dual_object.test_input, self.dual_object.true_class,
        self.dual_object.adv_class, self.dual_object.input_minval,
        self.dual_object.input_maxval, self.dual_object.epsilon)
    return projected_dual_object

  def tf_min_eig_vec(self):
    """Function for min eigen vector using tf's full eigen decomposition."""
    # Full eigen decomposition requires the explicit psd matrix M
    _, matrix_m = self.dual_object.get_full_psd_matrix()
    [eig_vals, eig_vectors] = tf.self_adjoint_eig(matrix_m)
    index = tf.argmin(eig_vals)
    return tf.reshape(
        eig_vectors[:, index], shape=[eig_vectors.shape[0].value, 1])

  def tf_smooth_eig_vec(self):
    """Function that returns smoothed version of min eigen vector."""
    _, matrix_m = self.dual_object.get_full_psd_matrix()
    # Easier to think in terms of max so negating the matrix
    [eig_vals, eig_vectors] = tf.self_adjoint_eig(-matrix_m)
    exp_eig_vals = tf.exp(tf.divide(eig_vals, self.smooth_placeholder))
    scaling_factor = tf.reduce_sum(exp_eig_vals)
    # Multiplying each eig vector by exponential of corresponding eig value
    # Scaling factor normalizes the vector to be unit norm
    eig_vec_smooth = tf.divide(
        tf.matmul(eig_vectors, tf.diag(tf.sqrt(exp_eig_vals))),
        tf.sqrt(scaling_factor))
    return tf.reshape(
        tf.reduce_sum(eig_vec_smooth, axis=1),
        shape=[eig_vec_smooth.shape[0].value, 1])

  def get_min_eig_vec_proxy(self, use_tf_eig=False):
    """Computes the min eigen value and corresponding vector of matrix M.

    Args:
      use_tf_eig: Whether to use tf's default full eigen decomposition

    Returns:
      eig_vec: Minimum absolute eigen value
      eig_val: Corresponding eigen vector
    """
    if use_tf_eig:
      # If smoothness parameter is too small, essentially no smoothing
      # Just output the eigen vector corresponding to min
      return tf.cond(self.smooth_placeholder < 1E-8,
                     self.tf_min_eig_vec,
                     self.tf_smooth_eig_vec)

    # Using autograph to automatically handle
    # the control flow of minimum_eigen_vector
    min_eigen_tf = autograph.to_graph(utils.minimum_eigen_vector)

    def _vector_prod_fn(x):
      return self.dual_object.get_psd_product(x)

    estimated_eigen_vector = min_eigen_tf(
        x=self.eig_init_vec_placeholder,
        num_steps=self.eig_num_iter_placeholder,
        learning_rate=self.params['eig_learning_rate'],
        vector_prod_fn=_vector_prod_fn)
    return estimated_eigen_vector

  def prepare_for_optimization(self):
    """Create tensorflow op for running one step of descent."""

    self.eig_vec_estimate = self.get_min_eig_vec_proxy()
    self.stopped_eig_vec_estimate = tf.stop_gradient(self.eig_vec_estimate)
    # Eig value is v^\top M v, where v is eigen vector
    self.eig_val_estimate = tf.matmul(
        tf.transpose(self.stopped_eig_vec_estimate),
        self.dual_object.get_psd_product(self.stopped_eig_vec_estimate))
    # Penalizing negative of min eigen value because we want min eig value
    # to be positive
    self.total_objective = (
        self.dual_object.unconstrained_objective
        + 0.5 * tf.square(
            tf.maximum(-self.penalty_placeholder * self.eig_val_estimate, 0)))
    global_step = tf.Variable(0, trainable=False)
    # Set up learning rate
    # Learning rate decays after every outer loop
    learning_rate = tf.train.exponential_decay(
        self.params['init_learning_rate'],
        global_step,
        self.params['inner_num_steps'],
        self.params['learning_rate_decay'],
        staircase=True)

    # Set up the optimizer
    if self.params['optimizer'] == 'adam':
      self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif self.params['optimizer'] == 'adagrad':
      self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif self.params['optimizer'] == 'momentum':
      self.optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=self.params['momentum_parameter'],
          use_nesterov=True)
    else:
      self.optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)

    # Write out the projection step
    self.train_step = self.optimizer.minimize(
        self.total_objective, global_step=global_step)
    # All dual variables are positive
    with tf.control_dependencies([self.train_step]):
      self.proj_step = tf.group(
          [v.assign(tf.maximum(v, 0)) for v in tf.trainable_variables()])

    # Control dependencies ensures that train_step is executed first
    self.opt_one_step = self.proj_step
    # Run the initialization of all variables
    # TODO: do we need to do it here or can do outside of this class?
    self.sess.run(tf.global_variables_initializer())
    # Create folder for saving stats if the folder is not None
    if (self.params.get('stats_folder') and
        not tf.gfile.IsDirectory(self.params['stats_folder'])):
      tf.gfile.MkDir(self.params['stats_folder'])

  def run_one_step(self, eig_init_vec_val, eig_num_iter_val, smooth_val,
                   penalty_val):
    """Run one step of gradient descent for optimization.

    Args:
      eig_init_vec_val: Start value for eigen value computations
      eig_num_iter_val: Number of iterations to run for eigen computations
      smooth_val: Value of smoothness parameter
      penalty_val: Value of penalty for the current step

    Returns:
     found_cert: True is negative certificate is found, False otherwise
    """
    # Project onto feasible set of dual variables
    if self.current_step % self.params['projection_steps'] == 0:
      current_certificate = self.sess.run(self.projected_dual_object.certificate)
      tf.logging.info('Inner step: %d, current value of certificate: %f',
                      self.current_step, current_certificate)

      # Sometimes due to either overflow or instability in inverses,
      # the returned certificate is large and negative -- keeping a check
      if LOWER_CERT_BOUND < current_certificate < 0:
        tf.logging.info('Found certificate of robustness!')
        return True
    # Running step
    step_feed_dict = {self.eig_init_vec_placeholder: eig_init_vec_val,
                      self.eig_num_iter_placeholder: eig_num_iter_val,
                      self.smooth_placeholder: smooth_val,
                      self.penalty_placeholder: penalty_val}

    [
        _, self.current_total_objective, self.current_unconstrained_objective,
        self.current_eig_vec_val, self.current_eig_val_estimate
    ] = self.sess.run([
        self.opt_one_step, self.total_objective,
        self.dual_object.unconstrained_objective, self.eig_vec_estimate,
        self.eig_val_estimate
    ], feed_dict=step_feed_dict)

    if self.current_step % self.params['print_stats_steps'] == 0:
      stats = {
          'total_objective':
              float(self.current_total_objective),
          'unconstrained_objective':
              float(self.current_unconstrained_objective),
          'min_eig_val_estimate':
              float(self.current_eig_val_estimate)
      }
      tf.logging.debug('Current inner step: %d, optimization stats: %s',
                       self.current_step, stats)
      if self.params['stats_folder'] is not None:
        stats = json.dumps(stats)
        filename = os.path.join(self.params['stats_folder'],
                                str(self.current_step) + '.json')
        with tf.gfile.Open(filename) as file_f:
          file_f.write(stats)
    return False

  def run_optimization(self):
    """Run the optimization, call run_one_step with suitable placeholders.

    Returns:
      True if certificate is found
      False otherwise
    """
    penalty_val = self.params['init_penalty']
    # Don't use smoothing initially - very inaccurate for large dimension
    self.smooth_on = False
    smooth_val = 0
    current_outer_step = 1
    while current_outer_step <= self.params['outer_num_steps']:
      tf.logging.info('Running outer step %d with penalty %f',
                      current_outer_step, penalty_val)
      # Running inner loop of optimization with current_smooth_val,
      # current_penalty as smoothness parameters and penalty respectively
      self.current_step = 0
      # Run first step with random eig initialization and large number of steps
      found_cert = self.run_one_step(
          np.random.random(size=(1 + self.dual_object.dual_index[-1], 1)),
          self.params['large_eig_num_steps'], smooth_val, penalty_val)
      if found_cert:
        return True
      while self.current_step < self.params['inner_num_steps']:
        self.current_step = self.current_step + 1
        found_cert = self.run_one_step(self.current_eig_vec_val,
                                       self.params['small_eig_num_steps'],
                                       smooth_val, penalty_val)
        if found_cert:
          return -1
      # Update penalty only if it looks like current objective is optimizes
      if self.current_total_objective < -0.2:
        penalty_val = penalty_val * self.params['beta']
      else:
        # To get more accurate gradient estimate
        self.params['small_eig_num_steps'] = (
            1.5 * self.params['small_eig_num_steps'])

      # If eigen values seem small enough, turn on smoothing
      # useful only when performing full eigen decomposition
      if np.abs(self.current_eig_val_estimate) < 0.01:
        smooth_val = self.params['smoothness_parameter']
      current_outer_step = current_outer_step + 1
    return False
