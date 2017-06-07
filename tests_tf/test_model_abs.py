from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from cleverhans.model_abs import Model, KerasModelWrapper


class TestModelClass(unittest.TestCase):
    def test_fprop(self):
        # Define empty model
        modelw = Model(model=None)
        x = []

        # Exception is thrown when `fprop` not implemented
        with self.assertRaises(Exception) as context:
            modelw.fprop(x, layer='')
        self.assertTrue(context.exception)

    def test_get_logits(self):
        # Define empty model
        modelw = Model(model=None)
        x = []

        # Exception is thrown when `fprop` not implemented
        with self.assertRaises(Exception) as context:
            modelw.get_logits(x)
        self.assertTrue(context.exception)

    def test_get_probs(self):
        # Define empty model
        modelw = Model(model=None)
        x = []

        # Exception is thrown when `fprop` not implemented
        with self.assertRaises(Exception) as context:
            modelw.get_probs(x)
        self.assertTrue(context.exception)


class TestKerasModelWrapper(unittest.TestCase):
    def test_fprop(self):
        from keras.models import Sequential
        from keras.layers import Dense
        import tensorflow as tf

        # Make a dummy Keras model with 2 dense layers
        input_shape = (100,)
        dummy_model = Sequential([Dense(20, name='l1',
                                        input_shape=input_shape),
                                  Dense(10, name='l2')])
        # Wrap Keras model
        modelw = KerasModelWrapper(dummy_model)
        # Get a symbolic representation for the hidden representation at l1
        x = tf.placeholder(tf.float32, shape=(None, 100))
        h1 = modelw.fprop(x, layer='l1')
        # Test the dimension of the hidden represetation
        self.assertEqual(int(h1.shape[1]), 20)


if __name__ == '__main__':
    unittest.main()
