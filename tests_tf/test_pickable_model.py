import numpy as np
import tensorflow as tf
from cleverhans.devtools.checks import CleverHansTest
from cleverhans.picklable_model import MLP
from cleverhans.picklable_model import PerImageStandardize


class TestPerImageStandardize(CleverHansTest):
  def setUp(self):
    super(TestPerImageStandardize, self).setUp()

    self.input_shape = (128, 32, 32, 3)
    self.sess = tf.Session()
    self.model = MLP(input_shape=self.input_shape,
                     layers=[PerImageStandardize(name='output')])

    self.x = tf.placeholder(shape=self.input_shape,
                            dtype=tf.float32)
    self.y = self.model.get_layer(self.x, 'output')

    self.y_true = tf.map_fn(
        lambda ex: tf.image.per_image_standardization(ex), self.x)

  def run_and_check_output(self, x):
    y, y_true = self.sess.run([self.y, self.y_true],
                              feed_dict={self.x: x})
    self.assertClose(y, y_true)

  def test_random_inputs(self):
    x = np.random.rand(*self.input_shape)
    self.run_and_check_output(x)

  def test_uniform_inputs(self):
    x = np.ones(self.input_shape)
    self.run_and_check_output(x)
