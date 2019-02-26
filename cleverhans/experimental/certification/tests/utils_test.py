"""Tests for cleverhans.experimental.certification.utils."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import autograph
from scipy.sparse.linalg import eigs, LinearOperator

from cleverhans.experimental.certification import utils

TOL = 1E-2
TEST_DIM = 50

class UtilsTest(tf.test.TestCase):

  def test_minimum_eigen_vector(self):
    matrix = np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32)
    initial_vec = np.array([[1.0], [-1.0]], dtype=np.float32)

    def _vector_prod_fn(x):
      return tf.matmul(matrix, x)

    min_eigen_fn = autograph.to_graph(utils.minimum_eigen_vector)
    x = tf.placeholder(tf.float32, shape=(2, 1))
    min_eig_vec = min_eigen_fn(x, 10, 0.1, _vector_prod_fn)
    with self.test_session() as sess:
      v = sess.run(min_eig_vec, feed_dict={x: initial_vec})
      if v.flatten()[0] < 0:
        v = -v
    np.testing.assert_almost_equal(v, [[0.9239], [-0.3827]], decimal=4)

  def basic_lanczos_test(self):
    # Define vector-product functions
    def _vector_prod_fn(x):
      return tf.matmul(matrix, x)
    def _np_vector_prod_fn(x):
      return np.matmul(matrix, x)

    # Create test diagonal matrix
    diagonal_entries = np.random.randint(low=0, size=TEST_DIM)
    matrix = np.diag(diagonal_entries)

    # Create lanczos graph nodes
    min_eigen_fn = autograph.to_graph(utils.lanczos_decomp)

    # Compare against scipy
    linear_operator = LinearOperator((TEST_DIM, TEST_DIM), matvec=_np_vector_prod_fn)
    min_eig_scipy, _ = eigs(linear_operator, k=1, which='SR', tol=TOL)
    print("Min eig scipy: " + str(min_eig_scipy))

    # Use lanczos method
    with tf.Session() as sess:
      alpha_hat, beta_hat, Q_hat = min_eigen_fn(_vector_prod_fn, 0, TEST_DIM, TEST_DIM/5)
      # Finalize graph to make sure no new nodes are added
      tf.get_default_graph().finalize()
      curr_alpha_hat, curr_beta_hat, _ = sess.run([alpha_hat, beta_hat, Q_hat])
      min_eig_lzs, _, _, _ = utils.eigen_tridiagonal(curr_alpha_hat, curr_beta_hat, maximum=False)
      print(min_eig_lzs)
      np.testing.assert_almost_equal(min_eig_lzs, min_eig_scipy, decimal=2)
    tf.reset_default_graph()

  def advanced_lanczos_test(self):
    k_vals = [25, 100]
    filenames = ['diverging.npy', 'regular.npy']

    def _vector_prod_fn(x):
      return tf.matmul(matrix, x)
    def _np_vector_prod_fn(x):
      return np.matmul(matrix, x)

    for filename in filenames:
      filename = './matrices/' + filename
      matrix = np.load(filename).astype(np.float32)
      n = matrix.shape[0]

      # Create lanczos graph nodes
      min_eigen_fn = autograph.to_graph(utils.lanczos_decomp)

      # Compare against scipy
      linear_operator = LinearOperator((n, n), matvec=_np_vector_prod_fn)
      min_eig_scipy, _ = eigs(linear_operator, k=1, which='SR', tol=TOL)
      print("Min eig scipy: " + str(min_eig_scipy))

      print('k\t\tlzs time\t\teigh time\t\tl_min err')

      for k in k_vals:
        # Use lanczos method
        with tf.Session() as sess:
          start = time.time()
          alpha_hat, beta_hat, Q_hat = min_eigen_fn(_vector_prod_fn, 0, n, k)
          # Finalize graph to make sure no new nodes are added
          tf.get_default_graph().finalize()
          curr_alpha_hat, curr_beta_hat, _ = sess.run([alpha_hat, beta_hat, Q_hat])
          lzs_time = time.time() - start
          start = time.time()
          min_eig_lzs, _, _, _ = utils.eigen_tridiagonal(curr_alpha_hat, curr_beta_hat, maximum=False)
          eigh_time = time.time() - start
          print(min_eig_lzs)
          print('%d\t\t%g\t\t%g\t\t%-10.5g\t\t%.5g' %(
              k, lzs_time, eigh_time, np.abs(min_eig_lzs-min_eig_scipy)/np.abs(min_eig_scipy),
              np.abs(min_eig_lzs-min_eig_scipy)))
          np.testing.assert_almost_equal(min_eig_lzs, min_eig_scipy, decimal=2)
        tf.reset_default_graph()

if __name__ == '__main__':
  tf.test.main()
