"""Tests for cleverhans.utils_tf"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from cleverhans import utils_tf
from cleverhans.devtools.checks import CleverHansTest


def numpy_kl_with_logits(p_logits, q_logits):
  def numpy_softmax(logits):
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

  p = numpy_softmax(p_logits)
  log_p = p_logits - np.log(np.sum(np.exp(p_logits), axis=1, keepdims=True))
  log_q = q_logits - np.log(np.sum(np.exp(q_logits), axis=1, keepdims=True))
  return (p * (log_p - log_q)).sum(axis=1).mean()


class TestUtilsTF(CleverHansTest):
  """Test class for utils_tf"""

  def setUp(self):
    super(TestUtilsTF, self).setUp()
    self.sess = tf.Session()

  def test_clip_by_value_numpy_dtype(self):
    # Test that it's possible to use clip_by_value while mixing numpy and tf
    clip_min = np.zeros((1,))
    clip_max = tf.ones((1,))
    x = tf.ones((1,))
    # The point of this test is just to make sure the casting logic doesn't raise an exception
    utils_tf.clip_by_value(x, clip_min, clip_max)

  def test_l2_batch_normalize(self):
    x = tf.random_normal((100, 1000))
    x_norm = self.sess.run(utils_tf.l2_batch_normalize(x))
    self.assertClose(np.sum(x_norm**2, axis=1), 1, atol=1e-6)

  def test_kl_with_logits(self):
    p_logits = tf.placeholder(tf.float32, shape=(100, 20))
    q_logits = tf.placeholder(tf.float32, shape=(100, 20))
    p_logits_np = np.random.normal(0, 10, size=(100, 20))
    q_logits_np = np.random.normal(0, 10, size=(100, 20))
    kl_div_tf = self.sess.run(utils_tf.kl_with_logits(p_logits, q_logits),
                              feed_dict={p_logits: p_logits_np,
                                         q_logits: q_logits_np})
    kl_div_ref = numpy_kl_with_logits(p_logits_np, q_logits_np)
    self.assertClose(kl_div_ref, kl_div_tf)

  def test_clip_eta_norm_0(self):
    """test_clip_eta_norm_0: Test that `clip_eta` still works when the
    norm of `eta` is zero. This used to cause a divide by zero for ord
    1 and ord 2."""
    eta = tf.zeros((5, 3))
    self.assertTrue(eta.dtype == tf.float32, eta.dtype)
    eps = .25
    for ord_arg in [np.inf, 1, 2]:
      try:
        clipped = utils_tf.clip_eta(eta, ord_arg, eps)
      except NotImplementedError:
        # Don't raise SkipTest, it skips the rest of the for loop
        continue
      clipped = self.sess.run(clipped)
      self.assertTrue(not np.any(np.isinf(clipped)))
      self.assertTrue(not np.any(np.isnan(clipped)), (ord_arg, clipped))

  def test_clip_eta_goldilocks(self):
    """test_clip_eta_goldilocks: Test that the clipping handles perturbations
    that are too small, just right, and too big correctly"""
    eta = tf.constant([[2.], [3.], [4.]])
    self.assertTrue(eta.dtype == tf.float32, eta.dtype)
    eps = 3.
    for ord_arg in [np.inf, 1, 2]:
      for sign in [-1., 1.]:
        try:
          clipped = utils_tf.clip_eta(eta * sign, ord_arg, eps)
        except NotImplementedError:
          # Don't raise SkipTest, it skips the rest of the for loop
          continue
        clipped_value = self.sess.run(clipped)
        gold = sign * np.array([[2.], [3.], [3.]])
        self.assertClose(clipped_value, gold)
        grad, = tf.gradients(clipped, eta)
        grad_value = self.sess.run(grad)
        # Note: the second 1. is debatable (the left-sided derivative
        # and the right-sided derivative do not match, so formally
        # the derivative is not defined). This test makes sure that
        # we at least handle this oddity consistently across all the
        # argument values we test
        gold = sign * np.array([[1.], [1.], [0.]])
        self.assertClose(grad_value, gold)

  def test_zero_out_clipped_grads(self):
    """
    test_zero_out_clipped_grads: Test that gradient gets zeroed out at positions
    where no progress can be made due to clipping.
    """

    clip_min = -1
    clip_max = 1
    eta = tf.constant([[0.], [-1.], [1], [0.5], [-1], [1], [-0.9], [0.9]])
    grad = tf.constant([[1.], [-1.], [1.], [1.], [1.], [-1.], [-1.], [1.]])

    grad2 = self.sess.run(
        utils_tf.zero_out_clipped_grads(grad, eta, clip_min, clip_max))

    expected = np.asarray([[1.], [0.], [0.], [1.], [1.], [-1.], [-1.], [1.]])
    self.assertClose(grad2, expected)

  def test_random_lp_vector_linf(self):
    """
    test_random_lp_sample_linf: Test that `random_lp_vector` returns
    random samples in the l-inf ball.
    """

    eps = 0.5
    d = 10

    r = self.sess.run(utils_tf.random_lp_vector((1000, d), np.infty, eps))

    # test that some values are close to the boundaries
    self.assertLessEqual(np.max(r), eps)
    self.assertGreaterEqual(np.max(r), 0.95*eps)
    self.assertGreaterEqual(np.min(r), -eps)
    self.assertLessEqual(np.min(r), -0.95*eps)

    # test that the mean value of each feature is close to zero
    means = np.mean(r, axis=0)
    self.assertClose(means, np.zeros(d), atol=0.05)

  def test_random_lp_srandom_lp_vector_l1_l2(self):
    """
    test_random_lp_vector_l1_l2: Test that `random_lp_vector` returns
    random samples in an l1 or l2 ball.
    """

    eps = 0.5
    d = 10

    for ord in [1, 2]:
      r = self.sess.run(utils_tf.random_lp_vector((1000, d), ord, eps))

      norms = np.linalg.norm(r, axis=-1, ord=ord)

      # test that some values are close to the boundaries
      self.assertLessEqual(np.max(norms), eps)
      self.assertGreaterEqual(np.max(norms), 0.95 * eps)

      # The expected norm is eps * Exp[U[0,1]^(1/d)] where U is a standard
      # uniform random variable and d is the dimension. The second term is
      # equal to the expected value of a Beta(d, 1) variable which is d/(d+1).
      expected_mean_norm = eps * (d / (d + 1.))
      self.assertClose(np.mean(norms), expected_mean_norm, atol=0.02)
