"""Tests for cleverhans.utils_tf"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nose.plugins.skip import SkipTest
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
