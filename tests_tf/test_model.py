from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from cleverhans.model import Model, KerasModelWrapper


class TestModelClass(unittest.TestCase):
    def test_default_graph_inference_state(self):
        # Define empty model
        model = Model(model=None)
        self.assertTrue(model.state == 'test')

    def test_change_graph_to_train(self):
        # Define empty model
        model = Model(model=None)

        # Set graph state to inference
        model.set_state('train')
        self.assertTrue(model.state == 'train')

    def test_fprop_layer(self):
        # Define empty model
        model = Model(model=None)
        x = []

        # Exception is thrown when `fprop_layer` not implemented
        with self.assertRaises(Exception) as context:
            model.fprop_layer(x, layer='')
        self.assertTrue(context.exception)

    def test_fprop_logits(self):
        # Define empty model
        model = Model(model=None)
        x = []

        # Exception is thrown when `fprop_logits` not implemented
        with self.assertRaises(Exception) as context:
            model.fprop_logits(x)
        self.assertTrue(context.exception)

    def test_fprop_probs(self):
        # Define empty model
        model = Model(model=None)
        x = []

        # Exception is thrown when `fprop_probs` not implemented
        with self.assertRaises(Exception) as context:
            model.fprop_probs(x)
        self.assertTrue(context.exception)

    def test_get_layer_names(self):
        # Define empty model
        model = Model(model=None)

        # Exception is thrown when `get_layer_names` not implemented
        with self.assertRaises(Exception) as context:
            model.get_layer_names()
        self.assertTrue(context.exception)

    def test_fprop(self):
        # Define empty model
        model = Model(model=None)
        x = []

        # Exception is thrown when `fprop` not implemented
        with self.assertRaises(Exception) as context:
            model.fprop(x)
        self.assertTrue(context.exception)


class TestKerasModelWrapper(unittest.TestCase):
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

    def test_set_state(self):
        model = KerasModelWrapper(self.model)
        # Exception is thrown when `set_train` is called
        with self.assertRaises(NotImplementedError) as context:
            model.set_state('train')
        self.assertTrue(context.exception)

    def test_get_softmax_name(self):
        model = KerasModelWrapper(self.model)
        softmax_name = model._get_softmax_name()
        self.assertEqual(softmax_name, 'softmax')

    def test_get_logits_name(self):
        model = KerasModelWrapper(self.model)
        logits_name = model._get_logits_name()
        self.assertEqual(logits_name, 'l2')

    def test_fprop_logits(self):
        import tensorflow as tf
        model = KerasModelWrapper(self.model)
        x = tf.placeholder(tf.float32, shape=(None, 100))
        preds = model.fprop_probs(x)
        logits = model.fprop_logits(x)

        x_val = np.random.rand(2, 100)
        tf.global_variables_initializer().run(session=self.sess)
        p_val, logits = self.sess.run([preds, logits], feed_dict={x: x_val})
        p_gt = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        self.assertTrue(np.allclose(p_val, p_gt, atol=1e-6))

    def test_fprop_probs(self):
        import tensorflow as tf
        model = KerasModelWrapper(self.model)
        x = tf.placeholder(tf.float32, shape=(None, 100))
        preds = model.fprop_probs(x)

        x_val = np.random.rand(2, 100)
        tf.global_variables_initializer().run(session=self.sess)
        p_val = self.sess.run(preds, feed_dict={x: x_val})
        self.assertTrue(np.allclose(np.sum(p_val, axis=1), 1, atol=1e-6))

    def test_get_layer_names(self):
        model = KerasModelWrapper(self.model)
        layer_names = model.get_layer_names()
        self.assertEqual(layer_names, ['l1', 'l2', 'softmax'])

    def test_fprop(self):
        import tensorflow as tf
        model = KerasModelWrapper(self.model)
        x = tf.placeholder(tf.float32, shape=(None, 100))
        out_dict = model.fprop(x)

        self.assertEqual(set(out_dict.keys()), set(['l1', 'l2', 'softmax']))
        # Test the dimension of the hidden represetation
        self.assertEqual(int(out_dict['l1'].shape[1]), 20)
        self.assertEqual(int(out_dict['l2'].shape[1]), 10)

        # Test the caching
        x2 = tf.placeholder(tf.float32, shape=(None, 100))
        out_dict2 = model.fprop(x2)
        self.assertEqual(set(out_dict2.keys()), set(['l1', 'l2', 'softmax']))
        self.assertEqual(int(out_dict2['l1'].shape[1]), 20)


if __name__ == '__main__':
    unittest.main()
