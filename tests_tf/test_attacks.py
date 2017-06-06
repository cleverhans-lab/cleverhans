from __future__ import absolute_import, division, print_function

import unittest
import numpy as np

import sys
sys.path = [".."]+sys.path
from cleverhans.attacks import VirtualAdversarialMethod, CarliniWagnerL2

"""
class TestVirtualAdversarialMethod:#(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        import tensorflow.contrib.slim as slim

        def dummy_model(x):
            net = slim.fully_connected(x, 600)
            return slim.fully_connected(net, 10, activation_fn=None)

        self.sess = tf.Session()
        self.sess.as_default()
        self.model = tf.make_template('dummy_model', dummy_model)
        self.attack = VirtualAdversarialMethod(self.model, sess=self.sess)

        # initialize model
        with tf.name_scope('dummy_model'):
            self.model(tf.placeholder(tf.float32, shape=(None, 1000)))
        self.sess.run(tf.initialize_all_variables())

    def test_parse_params(self):
        self.attack.parse_params()
        # test default values
        self.assertEqual(self.attack.eps, 2.0)
        self.assertEqual(self.attack.num_iterations, 1)
        self.assertEqual(self.attack.xi, 1e-6)
        self.assertEqual(self.attack.clip_min, None)
        self.assertEqual(self.attack.clip_max, None)

    def test_generate_np(self):
        x_val = np.random.rand(100, 1000)
        perturbation = self.attack.generate_np(x_val) - x_val
        perturbation_norm = np.sqrt(np.sum(perturbation**2, axis=1))
        # test perturbation norm
        self.assertTrue(np.allclose(perturbation_norm, self.attack.eps))
        """

class TestCarliniWagner(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        import tensorflow.contrib.slim as slim

        def dummy_model(x):
            net = slim.fully_connected(x, 60)
            return slim.fully_connected(net, 10, activation_fn=None)

        self.sess = tf.Session()
        self.sess.as_default()
        self.model = tf.make_template('dummy_modelq', dummy_model)
        self.attack = CarliniWagnerL2(self.model, sess=self.sess)

        # initialize model
        with tf.name_scope('dummy_modelq'):
            self.model(tf.placeholder(tf.float32, shape=(None, 100)))
        self.sess.run(tf.initialize_all_variables())
        
    def test_targeted_attack_returns_correct_target(self):
        x_val = np.random.rand(10, 100)
        y_val = np.zeros((10, 10))
        y_val[np.arange(10), np.random.random_integers(0,9,10)] = 1
        adversarial_example = self.attack.generate_np(x_val, y_val, batch_size=10,
                                                      initial_const=1, binary_search_steps=3)
        preds = self.sess.run(self.model(np.array(adversarial_example, dtype=np.float32)))
        print(np.argmax(preds,axis=1))
        print(np.argmax(y_val,axis=1))
        
        
        

if __name__ == '__main__':
    # t = TestVirtualAdversarialMethod("test_generate_np")
    # t.test_generate_np()
    unittest.main()
