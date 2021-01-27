"""Tests for cleverhans.picklable_model"""
import numpy as np
import tensorflow as tf

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.picklable_model import Dropout
from cleverhans.picklable_model import MLP
from cleverhans.picklable_model import PerImageStandardize


class TestPerImageStandardize(CleverHansTest):
  """
  Tests for the PerImageStandardize class.
  """

  def setUp(self):
    """
    Set up session and build model graph
    """
    super(TestPerImageStandardize, self).setUp()

    self.input_shape = (128, 32, 32, 3)
    self.sess = tf.Session()
    self.model = MLP(input_shape=self.input_shape,
                     layers=[PerImageStandardize(name='output')])

    self.x = tf.placeholder(shape=self.input_shape,
                            dtype=tf.float32)
    self.y = self.model.get_layer(self.x, 'output')

    self.y_true = tf.map_fn(tf.image.per_image_standardization, self.x)

  def run_and_check_output(self, x):
    """
    Make sure y and y_true evaluate to the same value
    """
    y, y_true = self.sess.run([self.y, self.y_true],
                              feed_dict={self.x: x})
    self.assertClose(y, y_true)

  def test_random_inputs(self):
    """
    Test on random inputs
    """
    x = np.random.rand(*self.input_shape)
    self.run_and_check_output(x)

  def test_ones_inputs(self):
    """
    Test with input set to all ones.
    """
    x = np.ones(self.input_shape)
    self.run_and_check_output(x)


class TestDropout(CleverHansTest):
  """
  Tests for the Dropout class
  """

  def test_no_drop(self):
    """test_no_drop: Make sure dropout does nothing by default
    (so it does not cause stochasticity at test time)"""

    model = MLP(input_shape=[1, 1], layers=[Dropout(name='output')])
    x = tf.constant([[1]], dtype=tf.float32)
    y = model.get_layer(x, 'output')
    sess = tf.Session()
    # Do multiple runs because dropout is stochastic
    for _ in range(10):
      y_value = sess.run(y)
      self.assertClose(y_value, 1.)

  def test_drop(self):
    """test_drop: Make sure dropout is activated successfully"""

    # We would like to configure the test to deterministically drop,
    # so that the test does not need to use multiple runs.
    # However, tf.nn.dropout divides by include_prob, so zero or
    # infinitesimal include_prob causes NaNs.
    # 1e-8 does not cause NaNs and shouldn't be a significant source
    # of test flakiness relative to dependency downloads failing, etc.
    model = MLP(input_shape=[1, 1], layers=[Dropout(name='output',
                                                    include_prob=1e-8)])
    x = tf.constant([[1]], dtype=tf.float32)
    y = model.get_layer(x, 'output', dropout=True)
    sess = tf.Session()
    y_value = sess.run(y)
    # Subject to very rare random failure because include_prob is not exact 0
    self.assertClose(y_value, 0.)

  def test_override(self):
    """test_override: Make sure dropout_dict changes dropout probabilities
    successfully."""

    # We would like to configure the test to deterministically drop,
    # so that the test does not need to use multiple runs.
    # However, tf.nn.dropout divides by include_prob, so zero or
    # infinitesimal include_prob causes NaNs.
    # For this test, random failure to drop will not cause the test to fail.
    # The stochastic version should not even run if everything is working
    # right.
    model = MLP(input_shape=[1, 1], layers=[Dropout(name='output',
                                                    include_prob=1e-8)])
    x = tf.constant([[1]], dtype=tf.float32)
    dropout_dict = {'output': 1.}
    y = model.get_layer(x, 'output', dropout=True, dropout_dict=dropout_dict)
    sess = tf.Session()
    y_value = sess.run(y)
    self.assertClose(y_value, 1.)
