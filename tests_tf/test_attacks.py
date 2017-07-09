from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.attacks import SaliencyMapMethod

import time

class CleverHansTest(unittest.TestCase):
    def setUp(self):
        self.test_start = time.time()
        
    def tearDown(self):
        print(self.id(), "took", time.time() - self.test_start, "seconds")


class TestVirtualAdversarialMethod(CleverHansTest):
    def setUp(self):
        super(TestVirtualAdversarialMethod, self).setUp()
        import tensorflow as tf
        import tensorflow.contrib.slim as slim

        def dummy_model(x):
            net = slim.fully_connected(x, 60)
            return slim.fully_connected(net, 10, activation_fn=None)

        self.sess = tf.Session()
        self.sess.as_default()
        self.model = tf.make_template('dummy_model', dummy_model)
        self.attack = VirtualAdversarialMethod(self.model, sess=self.sess)

        # initialize model
        with tf.name_scope('dummy_model'):
            self.model(tf.placeholder(tf.float32, shape=(None, 1000)))
        self.sess.run(tf.global_variables_initializer())

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

class TestFastGradientMethod(CleverHansTest):
    def setUp(self):
        super(TestFastGradientMethod, self).setUp()
        import tensorflow as tf

        # The world's simplest neural network
        def my_model(x):
            W1 = tf.constant([[1.5, .3], [-2, 0.3]], dtype=tf.float32)
            h1 = tf.nn.sigmoid(tf.matmul(x, W1))
            W2 = tf.constant([[-2.4, 1.2], [0.5, -2.3]], dtype=tf.float32)
            res = tf.nn.softmax(tf.matmul(x, W2))
            return res

        self.sess = tf.Session()
        self.model = my_model
        self.attack = FastGradientMethod(self.model, sess=self.sess)

    def test_generate_np_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        for ord in [np.inf, 1, 2]:
            x_adv = self.attack.generate_np(x_val, eps=.5, ord=ord,
                                            clip_min=-5, clip_max=5)

            if ord == np.inf:
                assert np.allclose(np.max(np.abs(x_adv-x_val), axis=1), 0.5)
            elif ord == 1:
                assert np.allclose(np.sum(np.abs(x_adv-x_val), axis=1), 0.5)
            elif ord == 2:
                assert np.allclose(np.sum(np.square(x_adv-x_val), axis=1)**.5,
                                   0.5)

            orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
            new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

            assert np.mean(orig_labs==new_labs) < 0.5

    def test_generate_np_can_be_called_with_different_eps(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        for eps in [0.1, 0.2, 0.3, 0.4]:
            x_adv = self.attack.generate_np(x_val, eps=eps, ord=np.inf,
                                            clip_min=-5.0, clip_max=5.0)

            assert np.allclose(np.max(np.abs(x_adv-x_val), axis=1), eps)

    def test_generate_np_clip_works_as_expected(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=0.5, ord=np.inf,
                                        clip_min=-0.2, clip_max=0.1)

        assert np.isclose(np.min(x_adv), -0.2)
        assert np.isclose(np.max(x_adv), 0.1)

    def test_generate_np_caches_graph_computation_for_eps_clip_or_xi(self):
        import tensorflow as tf

        x_val = np.random.rand(1, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=.3, num_iterations=10,
                                        clip_max=-5.0, clip_min=-5.0,
                                        xi=1e-6)

        old_grads = tf.gradients

        def fn(*x, **y):
            raise RuntimeError()
        tf.gradients = fn

        x_adv = self.attack.generate_np(x_val, eps=.2, num_iterations=10,
                                        clip_max=-4.0, clip_min=-4.0,
                                        xi=1e-5)
        
        tf.gradients = old_grads

    def test_generate_np_does_not_cache_graph_computation_for_num_iterations(self):
        import tensorflow as tf

        x_val = np.random.rand(1, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=.5, num_iterations=10,
                                        clip_min=-5.0, clip_max=5.0)

        ok = [False]
        old_grads = tf.gradients

        def fn(*x, **y):
            ok[0] = True
            return old_grads(*x, **y)
        tf.gradients = fn

        x_adv = self.attack.generate_np(x_val, eps=.5, num_iterations=20,
                                        clip_min=-5.0, clip_max=5.0)

        tf.gradients = old_grads

        assert ok[0]

class TestBasicIterativeMethod(TestFastGradientMethod):
    def setUp(self):
        super(TestBasicIterativeMethod, self).setUp()
        import tensorflow as tf

        # The world's simplest neural network
        def my_model(x):
            W1 = tf.constant([[1.5, .3], [-2, 0.3]], dtype=tf.float32)
            h1 = tf.nn.sigmoid(tf.matmul(x, W1))
            W2 = tf.constant([[-2.4, 1.2], [0.5, -2.3]], dtype=tf.float32)
            res = tf.nn.softmax(tf.matmul(x, W2))
            return res

        self.sess = tf.Session()
        self.model = my_model
        self.attack = BasicIterativeMethod(self.model, sess=self.sess)

    def test_generate_np_does_not_cache_graph_computation_for_nb_iter(self):
        import tensorflow as tf

        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, ord=np.inf,
                                        clip_min=-5.0, clip_max=5.0,
                                        nb_iter=10)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        assert np.mean(orig_labs==new_labs) < 0.1

        ok = [False]
        old_grads = tf.gradients

        def fn(*x, **y):
            ok[0] = True
            return old_grads(*x, **y)
        tf.gradients = fn

        x_adv = self.attack.generate_np(x_val, eps=1.0, ord=np.inf,
                                        clip_min=-5.0, clip_max=5.0,
                                        nb_iter=11)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        assert np.mean(orig_labs==new_labs) < 0.1

        tf.gradients = old_grads

        assert ok[0]


class TestSaliencyMapMethod(CleverHansTest):
    def setUp(self):
        super(TestSaliencyMapMethod, self).setUp()
        import tensorflow as tf
        import tensorflow.contrib.slim as slim

        def dummy_model(x):
            net = slim.fully_connected(x, 60)
            return slim.fully_connected(net, 10, activation_fn=None)

        self.sess = tf.Session()
        self.sess.as_default()
        self.model = tf.make_template('dummy_model', dummy_model)
        self.attack = VirtualAdversarialMethod(self.model, sess=self.sess)

        # initialize model
        with tf.name_scope('dummy_model'):
            self.model(tf.placeholder(tf.float32, shape=(None, 1000)))
        self.sess.run(tf.global_variables_initializer())

        self.attack = SaliencyMapMethod(self.model, sess=self.sess)

    def test_generate_np_targeted_gives_adversarial_example(self):
        x_val = np.random.rand(10, 1000)
        x_val = np.array(x_val, dtype=np.float32)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        feed_labs = np.zeros((10, 1000))
        feed_labs[np.arange(10), np.random.randint(0,9,10)] = 1
        x_adv = self.attack.generate_np(x_val,
                                        clip_min=-5, clip_max=5,
                                        targets=feed_labs, nb_classes=10)
        new_labs = self.sess.run(self.model(x_adv))
        
        assert np.mean(np.argmax(feed_labs,axis=1)==np.argmax(new_labs,axis=1)) == 1.0


if __name__ == '__main__':
    unittest.main()
