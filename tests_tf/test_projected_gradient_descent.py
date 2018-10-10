"""Tests for the ProjectGradientDescent attack
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nose.tools import assert_raises
import tensorflow as tf

from cleverhans.attacks import ProjectedGradientDescent

def test_callable_no_softmax():
  batch_size = 2
  nb_classes = 3
  def model(x):
    return tf.ones((batch_size, nb_classes)) / nb_classes
  sess = tf.Session()
  attack = ProjectedGradientDescent(model, sess=sess)
  x = tf.ones((batch_size, 3))
  # Currently ProjectedGradientDescent treats the output of a callable
  # as probs rather than logits.
  # Since our callable does not use a softmax, it is impossible to get
  # the logits back. The test confirms that this causes an error.
  assert_raises(TypeError, attack.generate, x)

if __name__ == "__main__":
  test_callable()
