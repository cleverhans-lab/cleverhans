"""Code for setting up the optimization problem for certification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time 
import numpy as np

from scipy.sparse.linalg import eigs, LinearOperator

import tensorflow as tf
from tensorflow.contrib import autograph
from cleverhans.experimental.certification import utils
from cleverhans.experimental.certification import dual_formulation

flags = tf.app.flags
FLAGS = flags.FLAGS

# Bound on lowest value of certificate to check for numerical errors
LOWER_CERT_BOUND = -10.0

class Optimization(object):
  """Class that sets up and runs the optimization of dual_formulation"""

  def __init__(self, dual_formulation_object, sess, optimization_params, nn_params_ff):
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
      nn_params_ff: Neural net param object with feedforward version 
    """
    self.dual_object = dual_formulation_object
    self.params = optimization_params
    self.penalty_placeholder = tf.placeholder(tf.float32, shape=[])

    # The dimensionality of matrix M is the sum of sizes of all layers + 1
    # The + 1 comes due to a row and column of M representing the linear terms
    self.eig_init_vec_placeholder = tf.placeholder(
        tf.float32, shape=[1 + self.dual_object.dual_index[-1], 1])
    self.smooth_placeholder = tf.placeholder(tf.float32, shape=[])
    self.eig_num_iter_placeholder = tf.placeholder(tf.int32, shape=[])
    self.sess = sess
    self.nn_params_ff = nn_params_ff
    self.dual_placeholder_ff = dual_formulation.DualFormulation(self.sess,
        nn_params_ff,
        self.dual_object.test_input,
        self.dual_object.true_class,
        self.dual_object.adv_class,
        self.dual_object.input_minval,
        self.dual_object.input_maxval,
        self.dual_object.epsilon
        )
    self.dual_placeholder_ff.initialize_placeholders()
    self.dual_placeholder_ff.get_full_psd_matrix()

  def tf_min_eig_vec(self):
    """Function for min eigen vector using tf's full eigen decomposition."""
    # Full eigen decomposition requires the explicit psd matrix M
    _, matrix_m = self.dual_object.get_full_psd_matrix()
    [eig_vals, eig_vectors] = tf.self_adjoint_eig(matrix_m)
    index = tf.argmin(eig_vals)
    return tf.reshape(eig_vectors[:, index],
                      shape=[eig_vectors.shape[0].value, 1])

  def tf_smooth_eig_vec(self):
    """Function that returns smoothed version of min eigen vector."""
    _, matrix_m = self.dual_object.get_full_psd_matrix()
    # Easier to think in terms of max so negating the matrix
    [eig_vals, eig_vectors] = tf.self_adjoint_eig(-matrix_m)
    exp_eig_vals = tf.exp(tf.divide(eig_vals, self.smooth_placeholder))
    scaling_factor = tf.reduce_sum(exp_eig_vals)
    # Multiplying each eig vector by exponential of corresponding eig value
    # Scaling factor normalizes the vector to be unit norm
    eig_vec_smooth = tf.divide(tf.matmul(eig_vectors,
                                         tf.diag(tf.sqrt(exp_eig_vals))),
                               tf.sqrt(scaling_factor))
    return tf.reshape(tf.reduce_sum(eig_vec_smooth, axis=1),
                      shape=[eig_vec_smooth.shape[0].value, 1])

  def get_min_eig_vec_proxy(self):
    """Computes the min eigen value and corresponding vector of matrix M.

    Args:
      use_tf_eig: Whether to use tf's default full eigen decomposition

    Returns:
      eig_vec: Minimum absolute eigen value
      eig_val: Corresponding eigen vector
    """
    if FLAGS.eig_type == 'TF_EIG':
      # If smoothness parameter is too small, essentially no smoothing
      # Just output the eigen vector corresponding to min
      return tf.cond(self.smooth_placeholder < 1E-8,
                     self.tf_min_eig_vec,
                     self.tf_smooth_eig_vec)

    elif FLAGS.eig_type == 'TF_GD':
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

  def get_scipy_eig_vec(self, eig_val_estimate=None):
    if FLAGS.eig_type == 'SCIPY_FF':
      self.dual_object.get_full_psd_matrix()
      matrix_m = self.sess.run(self.dual_object.matrix_m)
      min_eig_vec_val, estimated_eigen_vector = eigs(matrix_m, k=1, which='SR', 
        tol=1E-4, sigma=eig_val_estimate)
      return np.reshape(estimated_eigen_vector, [-1, 1]), min_eig_vec_val

    elif FLAGS.eig_type == 'SCIPY_CONV':
      dim = self.dual_object.matrix_m_dimension
      input_vector = tf.placeholder(tf.float32, shape=(dim, 1))
      output_vector = self.dual_object.get_psd_product(input_vector)

      def np_vector_prod_fn(np_vector):
        np_vector = np.reshape(np_vector, [-1, 1])
        output_np_vector = self.sess.run(output_vector, feed_dict={input_vector:np_vector})
        return output_np_vector
      linear_operator = LinearOperator((dim, dim), matvec=np_vector_prod_fn)
      # Performing shift invert scipy operation when eig val estimate is available
      if(eig_val_estimate):
        min_eig_vec_val, estimated_eigen_vector = eigs(linear_operator, 
          k=1, tol=1E-4, sigma=eig_val_estimate)
      else:
        min_eig_vec_val, estimated_eigen_vector = eigs(linear_operator, 
          k=1, which='SR', tol=1E-4)
      return np.reshape(estimated_eigen_vector, [-1, 1]), min_eig_vec_val

    else:      
      [current_lambda_pos, current_lambda_neg, current_lambda_quad, 
       current_lambda_lu, current_nu] = self.sess.run([self.dual_object.lambda_pos, 
                                                  self.dual_object.lambda_neg, 
                                                  self.dual_object.lambda_quad, 
                                                  self.dual_object.lambda_lu, 
                                                  self.dual_object.nu])
      lambda_feed_dict = {}
      lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in 
                               zip(self.dual_placeholder_ff.lambda_pos, current_lambda_pos)})
      lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in 
                               zip(self.dual_placeholder_ff.lambda_neg, current_lambda_neg)})
      lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in 
                               zip(self.dual_placeholder_ff.lambda_quad, current_lambda_quad)})
      lambda_feed_dict.update({lambda_placeholder:value for lambda_placeholder, value in 
                               zip(self.dual_placeholder_ff.lambda_lu, current_lambda_lu)})
      lambda_feed_dict.update({self.dual_placeholder_ff.nu: current_nu})
      matrix_m = self.sess.run(self.dual_placeholder_ff.matrix_m, feed_dict=lambda_feed_dict)
      if(eig_val_estimate):
        start = time.time()
        min_eig_val, estimated_eigen_vector = eigs(matrix_m, k=1, which = 'LR', 
                                                   tol=1E-4, sigma=eig_val_estimate)
        end = time.time()
      else:
        min_eig_val, estimated_eigen_vector = eigs(matrix_m, k=1, which='SR',
                                                   tol=1E-4)
      return np.reshape(estimated_eigen_vector, [-1, 1]), np.real(min_eig_val)




  def prepare_one_step(self):
    """Create tensorflow op for running one step of descent."""
    # Create the objective

    self.dual_object.set_differentiable_objective()
    if FLAGS.eig_type == 'TF_EIG' or FLAGS.eig_type =='TF_GD':
      self.eig_vec_estimate = self.get_min_eig_vec_proxy()
    # Eig vector estimate is "fed" into placeholder from scipy
    else:
      self.eig_vec_estimate = tf.placeholder(tf.float32, shape=(self.dual_object.matrix_m_dimension, 1))
    self.stopped_eig_vec_estimate = tf.stop_gradient(self.eig_vec_estimate)
    # Eig value is v^\top M v, where v is eigen vector
    self.eig_val_estimate = tf.matmul(tf.transpose(
        self.stopped_eig_vec_estimate),
                                      self.dual_object.get_psd_product(
                                          self.stopped_eig_vec_estimate))
    # Penalizing negative of min eigen value because we want min eig value
    # to be positive
    self.total_objective = (self.dual_object.unconstrained_objective +
                            0.5*(tf.square(
                                tf.maximum(-1*self.penalty_placeholder*
                                           self.eig_val_estimate, 0))))
    global_step = tf.Variable(0, trainable=False)
    # Set up learning rate as a placeholder
    self.learning_rate = tf.placeholder(tf.float32, shape=[])

    # Set up the optimizer
    if self.params['optimizer'] == 'adam':
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate)
    elif self.params['optimizer'] == 'adagrad':
      self.optimizer = tf.train.AdagradOptimizer(
        learning_rate=self.learning_rate)
    elif self.params['optimizer'] == 'momentum':
      self.optimizer = tf.train.MomentumOptimizer(
          learning_rate=self.learning_rate,
          momentum=self.params['momentum_parameter'], use_nesterov=True)
    else:
      self.optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=self.learning_rate)

    # Write out the projection step
    self.train_step = self.optimizer.minimize(self.total_objective,
                                              global_step=global_step)
    
    # Initialize dual variables 
    self.sess.run(tf.global_variables_initializer())

    # Projecting the dual variables
    proj_ops = []
    for i in range(self.dual_object.nn_params.num_hidden_layers + 1):
      # Lambda_pos ia non negative for switch indices, 
      # Unconstrained for positive indices 
      # Zero for negative indices 
      proj_ops.append(self.dual_object.lambda_pos[i].assign(
        tf.multiply(self.dual_object.positive_indices[i],
                    self.dual_object.lambda_pos[i])+
        tf.multiply(self.dual_object.switch_indices[i],
                    tf.nn.relu(self.dual_object.lambda_pos[i]))))
      proj_ops.append(self.dual_object.lambda_neg[i].assign(
        tf.multiply(self.dual_object.negative_indices[i],
                    self.dual_object.lambda_neg[i])+
        tf.multiply(self.dual_object.switch_indices[i],
                    tf.nn.relu(self.dual_object.lambda_neg[i]))))
      # Lambda_quad is only non zero and positive for switch 
      proj_ops.append(self.dual_object.lambda_quad[i].assign(
        tf.multiply(self.dual_object.switch_indices[i],
                    tf.nn.relu(self.dual_object.lambda_quad[i]))))
      # Lambda_lu is always non negative 
      proj_ops.append(self.dual_object.lambda_lu[i].assign(
        tf.nn.relu(self.dual_object.lambda_lu[i])))

    self.proj_step = tf.group(proj_ops)
    # Create folder for saving stats if the folder is not None
    if(self.params['stats_folder'] is not None and
       not tf.gfile.IsDirectory(self.params['stats_folder'])):
      tf.gfile.MkDir(self.params['stats_folder'])
    self.current_scipy_eig_val = None

  def run_one_step(self, eig_init_vec_val, eig_num_iter_val,
                   smooth_val, penalty_val, learning_rate_val):
    """Run one step of gradient descent for optimization.

    Args:
      eig_init_vec_val: Start value for eigen value computations
      eig_num_iter_val: Number of iterations to run for eigen computations
      smooth_val: Value of smoothness parameter
      penalty_val: Value of penalty for the current step
      learning_rate_val: Value of learning rate 

    Returns:
     found_cert: True is negative certificate is found, False otherwise
    """
    # Project onto feasible set of dual variables
    if self.current_step % self.params['projection_steps'] == 0 and False:

      projected_certificate = self.dual_object.project_dual()
      current_certificate = self.sess.run(projected_certificate)
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
                      self.penalty_placeholder: penalty_val, 
                      self.learning_rate: learning_rate_val}
     # Have to feed eig vec estimate if scipy eig computations 
    if FLAGS.eig_type == 'SCIPY_CONV' or FLAGS.eig_type == 'SCIPY_FF':
      self.current_eig_vector, self.current_eig_val_estimate = self.get_scipy_eig_vec(self.current_eig_val_estimate)
      step_feed_dict.update({self.eig_vec_estimate:current_eig_vector})

    self.sess.run(self.train_step, feed_dict=step_feed_dict)
    [_, self.current_eig_vec_val, 
     self.current_eig_val_estimate] = self.sess.run([self.proj_step, 
                                                     self.eig_vec_estimate, 
                                                   self.eig_val_estimate], 
                                                   feed_dict=step_feed_dict)

    if self.current_step % self.params['print_stats_steps'] == 0:
      [self.current_total_objective, self.current_unconstrained_objective,
       self.current_eig_vec_val,
       self.current_eig_val_estimate,
       self.current_nu] = self.sess.run(
         [self.total_objective,
          self.dual_object.unconstrained_objective,
          self.eig_vec_estimate,
          self.eig_val_estimate,
          self.dual_object.nu], feed_dict=step_feed_dict)
      start = time.time()
      # To reset the scipy_eig value estimate in case it has diverged
      if(not self.current_scipy_eig_val or self.current_step%1000 == 0):
        print("Computing full scipy value")
        self.current_eig_vec_val, self.current_scipy_eig_val = self.get_scipy_eig_vec(None)
      else:
        self.current_eig_vec_val, self.current_scipy_eig_val = self.get_scipy_eig_vec(self.current_scipy_eig_val - 0.1)
      end = time.time()
      print('Scipy time', end - start)
      stats = {'total_objective': float(self.current_total_objective),
               'unconstrained_objective': float(self.current_unconstrained_objective),
               'min_eig_val_estimate': float(self.current_eig_val_estimate),
               'current_nu': float(self.current_nu), 
               'scipy_min_eig_val': float(np.real(self.current_scipy_eig_val))}

      tf.logging.debug('Current inner step: %d, optimization stats: %s',
                       self.current_step, stats)
      if self.params['stats_folder'] is not None:
        stats = json.dumps(stats)
        with tf.gfile.Open((self.params['stats_folder']+ '/' +
                            str(self.current_step) + '.json'), mode='w')as file_f:
          file_f.write(stats)
        self.dual_object.save_dual(self.params['stats_folder'] + '/' + 
                                  str(self.current_outer_step), self.sess)
    return False

  def run_optimization(self):
    """Run the optimization, call run_one_step with suitable placeholders.

    Returns:
      True if certificate is found
      False otherwise
    """
    self.prepare_one_step()
    penalty_val = self.params['init_penalty']
    # Don't use smoothing initially - very inaccurate for large dimension
    self.smooth_on = False
    smooth_val = 0
    learning_rate_val = self.params['init_learning_rate']

    self.current_outer_step = 1
    while self.current_outer_step <= self.params['outer_num_steps']:
      tf.logging.info('Running outer step %d with penalty %f and learning rate %f',
                      self.current_outer_step, penalty_val, learning_rate_val)
      # Running inner loop of optimization with current_smooth_val,
      # current_penalty as smoothness parameters and penalty respectively
      self.current_step = 0
      # Run first step with random eig initialization and large number of steps
      found_cert = self.run_one_step(
          np.random.random(
              size=(1 + self.dual_object.dual_index[-1], 1)),
          self.params['large_eig_num_steps'],
          smooth_val,
          penalty_val, 
          learning_rate_val)
      if found_cert:
        return True
      while self.current_step < self.params['inner_num_steps']:
        self.current_step = self.current_step + 1
        found_cert = self.run_one_step(self.current_eig_vec_val,
                                       self.params['small_eig_num_steps'],
                                       smooth_val,
                                       penalty_val, 
                                       learning_rate_val)
        if found_cert:
          return -1
      # Update penalty only if it looks like current objective is optimizes
      if self.current_total_objective < -0.1:
        penalty_val = penalty_val*self.params['beta']
        learning_rate_val = learning_rate_val*self.params['learning_rate_decay']
      else:
        # To get more accurate gradient estimate
        self.params['small_eig_num_steps'] = (
            1.5*self.params['small_eig_num_steps'])

      # If eigen values seem small enough, turn on smoothing
      # useful only when performing full eigen decomposition
      if np.abs(self.current_eig_val_estimate) < 0.01:
        smooth_val = self.params['smoothness_parameter']
      self.current_outer_step = self.current_outer_step + 1
    return False
