"""Tests for cleverhans.picklable_model.PerImageStandardize"""
import numpy as np
import tensorflow as tf
from cleverhans.picklable_model import MLP
from cleverhans.picklable_model import PerImageStandardize


def test_per_image_standardize():
  """test_per_image_standardize: Make sure the outputs of PerImageStandardize
  are the same with tf.image.per_image_standardization"""

  input_shape = (128, 32, 32, 3)

  model = MLP(input_shape=input_shape,
              layers=[PerImageStandardize(name='output')])

  x = tf.random_normal(shape=input_shape)
  y = model.get_layer(x, 'output')

  y_true = tf.map_fn(
      lambda ex: tf.image.per_image_standardization(ex), x)

  sess = tf.Session()

  y_value, y_true_value = sess.run([y, y_true])

  assert np.allclose(y_value, y_true_value)
