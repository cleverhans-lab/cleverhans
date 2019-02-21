"""Tests for cleverhans.experimental.certification.utils."""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import autograph

from cleverhans.experimental.certification import utils


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


if __name__ == '__main__':
  tf.test.main()
