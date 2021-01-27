"""Tests for cleverhans.serial"""
import numpy as np
import tensorflow as tf

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.serial import PicklableVariable
from cleverhans.serial import load
from cleverhans.serial import save


class TestSerial(CleverHansTest):
  """
  Tests for cleverhans.serial
  """

  def test_save_and_load_var(self):
    """test_save_and_load_var: Test that we can save and load a
    PicklableVariable with joblib
    """
    sess = tf.Session()
    with sess.as_default():
      x = np.ones(1)
      xv = PicklableVariable(x)
      xv.var.initializer.run()
      save("/tmp/var.joblib", xv)
      sess.run(tf.assign(xv.var, np.ones(1) * 2))
      new_xv = load("/tmp/var.joblib")
      self.assertClose(sess.run(xv.var), np.ones(1) * 2)
      self.assertClose(sess.run(new_xv.var), np.ones(1))
