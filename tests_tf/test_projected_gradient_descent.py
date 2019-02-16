"""Tests for the ProjectGradientDescent attack
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nose.tools import assert_raises
import tensorflow as tf

from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.model import Model


def test_no_logits():
  """test_no_logits: Check that a model without logits causes an error"""
  batch_size = 2
  nb_classes = 3

  class NoLogitsModel(Model):
    """
    A model that neither defines logits nor makes it possible to find logits
    by inspecting the inputs to a softmax op.
    """

    def fprop(self, x, **kwargs):
      return {'probs': tf.ones((batch_size, nb_classes)) / nb_classes}
  model = NoLogitsModel()
  sess = tf.Session()
  attack = ProjectedGradientDescent(model, sess=sess)
  x = tf.ones((batch_size, 3))
  assert_raises(NotImplementedError, attack.generate, x)


def test_rejects_callable():
  """test_rejects_callable: Check that callables are not accepted as models"""
  def model(x):
    """Mock model"""
    return x
  sess = tf.Session()
  assert_raises(TypeError, ProjectedGradientDescent, model, sess)


if __name__ == "__main__":
  test_rejects_callable()
