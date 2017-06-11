from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from cleverhans.exposer import ModelExposer, KerasModelExposer


class TestModelExposer(unittest.TestCase):
    def test_expose_layer(self):
        # Define empty model
        exposer = ModelExposer(model=None)

        # Exception is thrown when `expose_layer` not implemented
        with self.assertRaises(Exception) as context:
            exposer.expose_layer(layer='')
        self.assertTrue(context.exception)

    def test_expose_logits(self):
        # Define empty model
        exposer = ModelExposer(model=None)

        # Exception is thrown when `expose_logits` not implemented
        with self.assertRaises(Exception) as context:
            exposer.expose_logits()
        self.assertTrue(context.exception)

    def test_expose_probs(self):
        # Define empty model
        exposer = ModelExposer(model=None)

        # Exception is thrown when `expose_probs` not implemented
        with self.assertRaises(Exception) as context:
            exposer.expose_probs()
        self.assertTrue(context.exception)

    def test_get_layer_names(self):
        # Define empty model
        exposer = ModelExposer(model=None)

        # Exception is thrown when `get_layer_names` not implemented
        with self.assertRaises(Exception) as context:
            exposer.get_layer_names()
        self.assertTrue(context.exception)

    def test_expose_all_layers(self):
        # Define empty model
        exposer = ModelExposer(model=None)

        # Exception is thrown when `expose_all_layers` not implemented
        with self.assertRaises(Exception) as context:
            exposer.expose_all_layers()
        self.assertTrue(context.exception)


class TestKerasModelExposer(unittest.TestCase):
    def setUp(self):
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        import tensorflow as tf

        def dummy_model():
            input_shape = (100,)
            return Sequential([Dense(20, name='l1',
                                     input_shape=input_shape),
                               Dense(10, name='l2'),
                               Activation('softmax', name='softmax')])

        self.sess = tf.Session()
        self.sess.as_default()
        self.model = dummy_model()

    def test_expose_layer(self):
        import tensorflow as tf
        exposer = KerasModelExposer(self.model)
        x = tf.placeholder(tf.float32, shape=(None, 100))
        x2 = tf.placeholder(tf.float32, shape=(None, 100))
        model_expd = exposer.expose_layer(layer='l1')
        h1 = model_expd(x)
        h2 = model_expd(x2)

        # Test the dimension of the hidden represetation
        self.assertEqual(int(h1.shape[1]), 20)
        # Test the caching
        self.assertEqual(int(h2.shape[1]), 20)

    def test_probs(self):
        import tensorflow as tf
        exposer = KerasModelExposer(self.model)
        x = tf.placeholder(tf.float32, shape=(None, 100))
        model_slice = exposer.expose_probs()
        preds = model_slice(x)

        x_val = np.random.rand(2, 100)
        tf.global_variables_initializer().run(session=self.sess)
        p_val = self.sess.run(preds, feed_dict={x: x_val})
        self.assertTrue(np.allclose(np.sum(p_val, axis=1), 1, atol=1e-6))

    def test_logits(self):
        import tensorflow as tf
        exposer = KerasModelExposer(self.model)
        x = tf.placeholder(tf.float32, shape=(None, 100))
        model_slice = exposer.expose_probs()
        logits_slice = exposer.expose_logits()
        preds = model_slice(x)
        logits = logits_slice(x)

        x_val = np.random.rand(2, 100)
        tf.global_variables_initializer().run(session=self.sess)
        p_val, logits = self.sess.run([preds, logits], feed_dict={x: x_val})
        p_gt = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        self.assertTrue(np.allclose(p_val, p_gt, atol=1e-6))

    def test_get_layer_names(self):
        exposer = KerasModelExposer(self.model)
        layer_names = exposer.get_layer_names()
        self.assertEqual(layer_names, ['l1', 'l2', 'softmax'])

    def test_expose_all_layers(self):
        import tensorflow as tf
        exposer = KerasModelExposer(self.model)
        x = tf.placeholder(tf.float32, shape=(None, 100))
        model_expd = exposer.expose_all_layers()
        feats = model_expd(x)

        # Test the dimension of the hidden represetation
        self.assertEqual(int(feats[0].shape[1]), 20)
        self.assertEqual(int(feats[1].shape[1]), 10)

        # Test the caching
        x2 = tf.placeholder(tf.float32, shape=(None, 100))
        feats2 = model_expd(x2)
        self.assertEqual(int(feats2[0].shape[1]), 20)


if __name__ == '__main__':
    unittest.main()
