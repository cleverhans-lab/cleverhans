"""
This file defines code for eigendecomposition.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import autograph
from cleverhans.experimental.certification import utils
from cleverhans.experimental.certification import dual_formulation

class EigenDecomposition(object):

  def __init__(self, dual_object, learning_rate):
    self.dual_object = dual_object

    # The dimensionality of matrix M is the sum of sizes of all layers + 1
    # The + 1 comes due to a row and column of M representing the linear terms
    self.penalty_placeholder = tf.placeholder(tf.float32, shape=[])
    self.eig_init_vec_placeholder = tf.placeholder(
        tf.float32, shape=[1 + self.dual_object.dual_index[-1], 1])
    self.smooth_placeholder = tf.placeholder(tf.float32, shape=[])
    self.eig_num_iter_placeholder = tf.placeholder(tf.int32, shape=[])
    self.learning_rate = learning_rate

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
    # Scaling factor normalizes the vector to be unit normelf.params['eig_learning_rate']
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
      return tf.cond(self.smooth_placeholder < 1E-8, self.tf_min_eig_vec,
                     self.tf_smooth_eig_vec)

    # Using autograph to automatically handle
    # the control flow of minimum_eigen_vector
    min_eigen_tf = autograph.to_graph(utils.minimum_eigen_vector)

    def _vector_prod_fn(x):
      return self.dual_object.get_psd_product(x)

    estimated_eigen_vector = min_eigen_tf(
        x=self.eig_init_vec_placeholder,
        num_steps=self.eig_num_iter_placeholder,
        learning_rate=self.learning_rate,
        vector_prod_fn=_vector_prod_fn)
    return estimated_eigen_vector
  
  def create_estimates (self):
    self.eig_vec_estimate = self.get_min_eig_vec_proxy()
    self.stopped_eig_vec_estimate = tf.stop_gradient(self.eig_vec_estimate)
    # Eig value is v^\top M v, where v is eigen vector
    self.eig_val_estimate = tf.matmul(
        tf.transpose(self.stopped_eig_vec_estimate),
        self.dual_object.get_psd_product(self.stopped_eig_vec_estimate))
  
  def create_feed_dict (self, eig_init_vec_val, eig_num_iter_val, smooth_val, penalty_val):
    step_feed_dict = {
        self.eig_init_vec_placeholder: eig_init_vec_val,
        self.eig_num_iter_placeholder: eig_num_iter_val,
        self.smooth_placeholder: smooth_val,
        self.penalty_placeholder: penalty_val
    }
    return step_feed_dict
