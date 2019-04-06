"""Tests for cleverhans.experimental.certification.utils."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse.linalg import eigs
import tensorflow as tf
from tensorflow.contrib import autograph

from cleverhans.experimental.certification import utils

MATRIX_DIMENTION = 100
NUM_LZS_ITERATIONS = 100
NUM_RANDOM_MATRICES = 10


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

  def test_tf_lanczos_smallest_eigval(self):
    tf_num_iter = tf.placeholder(dtype=tf.int32, shape=())
    tf_matrix = tf.placeholder(dtype=tf.float32)
    def _vector_prod_fn(x):
      return tf.matmul(tf_matrix, tf.reshape(x, [-1, 1]))

    min_eigen_fn = autograph.to_graph(utils.tf_lanczos_smallest_eigval)
    init_vec_ph = tf.placeholder(shape=(MATRIX_DIMENTION, 1), dtype=tf.float32)
    tf_eigval, tf_eigvec = min_eigen_fn(
        _vector_prod_fn, MATRIX_DIMENTION, init_vec_ph, tf_num_iter, dtype=tf.float32)
    eigvec = np.zeros((MATRIX_DIMENTION, 1), dtype=np.float32)

    with self.test_session() as sess:
      # run this test for a few random matrices
      for _ in range(NUM_RANDOM_MATRICES):
        matrix = np.random.random((MATRIX_DIMENTION, MATRIX_DIMENTION))
        matrix = matrix + matrix.T  # symmetrizing matrix
        eigval, eigvec = sess.run(
            [tf_eigval, tf_eigvec],
            feed_dict={tf_num_iter: NUM_LZS_ITERATIONS, tf_matrix: matrix, init_vec_ph: eigvec})

        scipy_min_eigval, scipy_min_eigvec = eigs(
            matrix, k=1, which='SR')
        scipy_min_eigval = np.real(scipy_min_eigval)
        scipy_min_eigvec = np.real(scipy_min_eigvec)
        scipy_min_eigvec = scipy_min_eigvec / np.linalg.norm(scipy_min_eigvec)

        np.testing.assert_almost_equal(eigval, scipy_min_eigval, decimal=3)
        np.testing.assert_almost_equal(np.linalg.norm(eigvec), 1.0, decimal=3)
        abs_dot_prod = abs(np.dot(eigvec.flatten(), scipy_min_eigvec.flatten()))
        np.testing.assert_almost_equal(abs_dot_prod, 1.0, decimal=3)


if __name__ == '__main__':
  tf.test.main()
