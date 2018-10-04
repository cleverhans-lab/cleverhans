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

  x = tf.placeholder(shape=input_shape, dtype=tf.float32)
  y = model.get_layer(x, 'output')

  y_true = tf.map_fn(
      lambda ex: tf.image.per_image_standardization(ex), x)

  sess = tf.Session()

  # Random values
  x_random = np.random.rand(*input_shape)
  y_value, y_true_value = sess.run([y, y_true], feed_dict={x: x_random})
  assert np.allclose(y_value, y_true_value)

  # Uniform values to make zero variance
  x_uniform = np.ones(input_shape)
  y_value, y_true_value = sess.run([y, y_true], feed_dict={x: x_uniform})
  assert np.allclose(y_value, y_true_value)
