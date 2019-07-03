"""Tests for attacks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from cleverhans.future.tf2.attacks.fast_gradient_method import fast_gradient_method


class SimpleModel(tf.keras.Model):
  def __init__(self):
    super(SimpleModel, self).__init__()
    self.w1 = tf.constant([[1.5, .3], [-2, .3]])
    self.w2 = tf.constant([[-2.4, 1.2], [.5, -2.3]])

  def call(self, x):
    x = tf.linalg.matmul(x, self.w1)
    x = tf.math.sigmoid(x)
    x = tf.linalg.matmul(x, self.w2)
    return x


class CommonAttackProperties(tf.test.TestCase):

  def setUp(self):
    super(CommonAttackProperties, self).setUp()
    self.model = SimpleModel()
    self.x = tf.random.uniform((100, 2))
    self.normalized_x = tf.random.uniform((100, 2))  # truncated between [0, 1)
    self.red_ind = list(range(1, self.x.shape[0]))
    self.ord_list = [1, 2, np.inf]

  def help_adv_examples_success_rate(self, **kwargs):
    x_adv = self.attack(model_fn=self.model, x=self.normalized_x, **kwargs)
    ori_label = tf.math.argmax(self.model(self.normalized_x), -1)
    adv_label = tf.math.argmax(self.model(x_adv), -1)
    adv_acc = tf.math.reduce_mean(
        tf.cast(tf.math.equal(adv_label, ori_label), tf.float32))
    self.assertLess(adv_acc, .5)

  def help_targeted_adv_examples_success_rate(self, **kwargs):
    y_target = tf.random.uniform(shape=(self.normalized_x.shape[0],),
                                 minval=0, maxval=2, dtype=tf.int64)
    x_adv = self.attack(model_fn=self.model, x=self.normalized_x,
                        y=y_target, targeted=True, **kwargs)
    adv_label = tf.math.argmax(self.model(x_adv), -1)
    adv_success = tf.math.reduce_mean(
        tf.cast(tf.math.equal(adv_label, y_target), tf.float32))
    self.assertGreater(adv_success, .7)


class TestFastGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestFastGradientMethod, self).setUp()
    self.attack = fast_gradient_method
    self.eps_list = [0, .1, .3, 1., 3]
    self.attack_param = {
        'eps': .5,
        'clip_min': -5,
        'clip_max': 5
    }

  def test_invalid_input(self):
    x = tf.constant([[-2., 3.]])
    for norm in self.ord_list:
      with self.assertRaises(AssertionError):
        self.attack(model_fn=self.model, x=x, eps=.1, norm=norm,
                    clip_min=-1., clip_max=1., sanity_checks=True)

  def test_invalid_eps(self):
    for norm in self.ord_list:
      with self.assertRaises(ValueError):
        self.attack(model_fn=self.model, x=self.x, eps=-.1, norm=norm)

  def test_eps_equals_zero(self):
    for norm in self.ord_list:
      self.assertAllClose(self.attack(model_fn=self.model, x=self.x, eps=0, norm=norm),
                          self.x)

  def test_eps(self):
    # test if the attack respects the norm constraint
    # NOTE this has been tested with the optimize_linear function in
    # test_utils, so duplicate tests are not needed here.
    # Although, if ever switch the engine of the FGM method to some
    # function other than optimize_linear. This test should be added.
    raise self.skipTest("TODO")

  def test_clips(self):
    clip_min = -1.
    clip_max = 1.
    for norm in self.ord_list:
      x_adv = self.attack(
          model_fn=self.model, x=self.normalized_x, eps=.3, norm=norm,
          clip_min=clip_min, clip_max=clip_max)
      self.assertAllLessEqual(x_adv, clip_max)
      self.assertAllGreaterEqual(x_adv, clip_min)

  def test_invalid_clips(self):
    clip_min = .5
    clip_max = -.5
    for norm in self.ord_list:
      with self.assertRaises(ValueError):
        self.attack(model_fn=self.model, x=self.x, eps=.1, norm=norm,
                    clip_min=clip_min, clip_max=clip_max)

  def test_adv_example_success_rate_linf(self):
    # use normalized_x to make sure the same eps gives uniformly high attack
    # success rate across randomized tests
    self.help_adv_examples_success_rate(
        norm=np.inf, **self.attack_param)

  def test_targeted_adv_example_success_rate_linf(self):
    self.help_targeted_adv_examples_success_rate(
        norm=np.inf, **self.attack_param)

  def test_adv_example_success_rate_l1(self):
    self.help_adv_examples_success_rate(
        norm=1, **self.attack_param)

  def test_targeted_adv_example_success_rate_l1(self):
    self.help_targeted_adv_examples_success_rate(
        norm=1, **self.attack_param)

  def test_adv_example_success_rate_l2(self):
    self.help_adv_examples_success_rate(
        norm=2, **self.attack_param)

  def test_targeted_adv_example_success_rate_l2(self):
    self.help_targeted_adv_examples_success_rate(
        norm=2, **self.attack_param)


if __name__ == "__main__":
  tf.test.main()
