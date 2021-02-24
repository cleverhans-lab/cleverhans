"""
Tests for cleverhans.model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from cleverhans.model import Model, CallableModelWrapper


class TestModelClass(unittest.TestCase):
  """
  Tests for cleverhans.model.Model
  """
  # pylint: disable=missing-docstring

  def test_get_logits(self):
    # Define empty model
    model = Model('model', 10, {})
    x = []

    # Exception is thrown when `get_logits` not implemented
    with self.assertRaises(Exception) as context:
      model.get_logits(x)
    self.assertTrue(context.exception)

  def test_get_probs(self):
    # Define empty model
    model = Model('model', 10, {})
    x = []

    # Exception is thrown when `get_probs` not implemented
    with self.assertRaises(Exception) as context:
      model.get_probs(x)
    self.assertTrue(context.exception)

  def test_fprop(self):
    # Define empty model
    model = Model('model', 10, {})
    x = []

    # Exception is thrown when `fprop` not implemented
    with self.assertRaises(Exception) as context:
      model.fprop(x)
    self.assertTrue(context.exception)


class TestCallableModelWrapperInitArguments(unittest.TestCase):
  """
  Tests for CallableModelWrapper's init argument
  """

  def test_output_layer(self):
    """
    Test that the CallableModelWrapper can be constructed without causing Exceptions
    """
    def model(**kwargs):
      """Mock model"""
      del kwargs
      return True

    # The following two calls should not raise Exceptions
    CallableModelWrapper(model, 'probs')
    CallableModelWrapper(model, 'logits')


if __name__ == '__main__':
  unittest.main()
