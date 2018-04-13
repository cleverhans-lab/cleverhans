from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import tensorflow.contrib.slim as slim
import unittest
import numpy as np

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.attacks import Attack
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import FastFeatureAdversaries
from cleverhans.attacks import LBFGS
from cleverhans.model import Model

class SimpleModel(Model):
    """
    A very simple neural network
    """

    def get_logits(self, x):
        W1 = tf.constant([[1.5, .3], [-2, 0.3]], dtype=tf.as_dtype(x.dtype))
        h1 = tf.nn.sigmoid(tf.matmul(x, W1))
        W2 = tf.constant([[-2.4, 1.2], [0.5, -2.3]], dtype=tf.as_dtype(x.dtype))

        res = tf.matmul(h1, W2)
        return res

class TrivialModel(Model):
  """
  A linear model with two weights
  """

  def get_logits(self, x):
      W1 = tf.constant([[1, -1]], dtype=tf.float32)
      res = tf.matmul(x, W1)
      return res


class DummyModel(Model):
    """
    A simple model based on slim
    """

    def __init__(self):
        def template_fn(x):
          net = slim.fully_connected(x, 60)
          return slim.fully_connected(net, 10, activation_fn=None)
        self.template = tf.make_template('dummy_model', template_fn)

    def get_logits(self, x):
        return self.template(x)

class TestAttackClassInitArguments(CleverHansTest):

    def test_model(self):
        sess = tf.Session()

        # Exception is thrown when model does not have __call__ attribute
        with self.assertRaises(Exception) as context:
            model = tf.placeholder(tf.float32, shape=(None, 10))
            Attack(model, back='tf', sess=sess)
        self.assertTrue(context.exception)

    def test_back(self):
        model = Model()

        # Exception is thrown when back is not tf or th
        with self.assertRaises(Exception) as context:
            Attack(model, back='test', sess=None)
        self.assertTrue(context.exception)

    def test_sess(self):
        # Define empty model
        model = Model()

        # Test that it is permitted to provide no session
        Attack(model, back='tf', sess=None)

    def test_sess_generate_np(self):
        model = Model()

        class DummyAttack(Attack):
            def generate(self, x, **kwargs):
                return x

        attack = DummyAttack(model, back='tf', sess=None)
        with self.assertRaises(Exception) as context:
            attack.generate_np(0.)
        self.assertTrue(context.exception)


class TestParseParams(CleverHansTest):
    def test_parse(self):
        sess = tf.Session()

        test_attack = Attack(Model(), back='tf', sess=sess)
        self.assertTrue(test_attack.parse_params({}))


class TestVirtualAdversarialMethod(CleverHansTest):
    def setUp(self):
        super(TestVirtualAdversarialMethod, self).setUp()

        self.sess = tf.Session()
        self.sess.as_default()
        self.model = DummyModel()
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
        self.assertClose(perturbation_norm, self.attack.eps)


class TestFastGradientMethod(CleverHansTest):
    def setUp(self):
        super(TestFastGradientMethod, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = FastGradientMethod(self.model, sess=self.sess)

    def generate_adversarial_examples_np(self, ord, eps, **kwargs):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=eps, ord=ord,
                                        clip_min=-5, clip_max=5, **kwargs)
        if ord == np.inf:
            delta = np.max(np.abs(x_adv - x_val), axis=1)
        elif ord == 1:
            delta = np.sum(np.abs(x_adv - x_val), axis=1)
        elif ord == 2:
            delta = np.sum(np.square(x_adv - x_val), axis=1)**.5

        return x_val, x_adv, delta

    def help_generate_np_gives_adversarial_example(self, ord, eps=.5, **kwargs):
        x_val, x_adv, delta = self.generate_adversarial_examples_np(ord, eps,
                                                                    **kwargs)
        self.assertClose(delta, eps)
        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        self.assertTrue(np.mean(orig_labs == new_labs) < 0.5)

    def test_generate_np_gives_adversarial_example_linfinity(self):
        self.help_generate_np_gives_adversarial_example(np.infty)

    def test_generate_np_gives_adversarial_example_l1(self):
        self.help_generate_np_gives_adversarial_example(1)

    def test_generate_np_gives_adversarial_example_l2(self):
        self.help_generate_np_gives_adversarial_example(2)

    def test_generate_respects_dtype(self):
        x = tf.placeholder(dtype=tf.float64, shape=(100, 2))
        x_adv = self.attack.generate(x)
        self.assertEqual(x_adv.dtype, tf.float64)

    def test_targeted_generate_np_gives_adversarial_example(self):
        random_labs = np.random.random_integers(0, 1, 100)
        random_labs_one_hot = np.zeros((100, 2))
        random_labs_one_hot[np.arange(100), random_labs] = 1

        _, x_adv, delta = self.generate_adversarial_examples_np(
            eps=.5, ord=np.inf, y_target=random_labs_one_hot)

        self.assertClose(delta, 0.5)

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        self.assertTrue(np.mean(random_labs == new_labs) > 0.7)

    def test_generate_np_can_be_called_with_different_eps(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        for eps in [0.1, 0.2, 0.3, 0.4]:
            x_adv = self.attack.generate_np(x_val, eps=eps, ord=np.inf,
                                            clip_min=-5.0, clip_max=5.0)

            delta = np.max(np.abs(x_adv - x_val), axis=1)
            self.assertClose(delta, eps)

    def test_generate_np_clip_works_as_expected(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=0.5, ord=np.inf,
                                        clip_min=-0.2, clip_max=0.1)

        self.assertClose(np.min(x_adv), -0.2)
        self.assertClose(np.max(x_adv), 0.1)

    def test_generate_np_caches_graph_computation_for_eps_clip_or_xi(self):

        x_val = np.random.rand(1, 2)
        x_val = np.array(x_val, dtype=np.float32)

        self.attack.generate_np(x_val, eps=.3, num_iterations=10,
                                clip_max=-5.0, clip_min=-5.0,
                                xi=1e-6)

        old_grads = tf.gradients

        def fn(*x, **y):
            raise RuntimeError()
        tf.gradients = fn

        self.attack.generate_np(x_val, eps=.2, num_iterations=10,
                                clip_max=-4.0, clip_min=-4.0,
                                xi=1e-5)

        tf.gradients = old_grads


class TestBasicIterativeMethod(TestFastGradientMethod):
    def setUp(self):
        super(TestBasicIterativeMethod, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = BasicIterativeMethod(self.model, sess=self.sess)

    def test_generate_np_gives_adversarial_example_linfinity(self):
        self.help_generate_np_gives_adversarial_example(ord=np.infty, eps=.5,
                                                        nb_iter=20)

    def test_generate_np_gives_adversarial_example_l1(self):
        self.help_generate_np_gives_adversarial_example(ord=1, eps=.5,
                                                        nb_iter=20)

    def test_generate_np_gives_adversarial_example_l2(self):
        self.help_generate_np_gives_adversarial_example(ord=2, eps=.5,
                                                        nb_iter=20)

    def test_do_not_reach_lp_boundary(self):
        """
        Make sure that iterative attack don't reach boundary of Lp
        neighbourhood if nb_iter * eps_iter is relatively small compared to
        epsilon.
        """
        for ord in [1, 2, np.infty]:
            _, _, delta = self.generate_adversarial_examples_np(
                ord=ord, eps=.5, nb_iter=10, eps_iter=.01)
            self.assertTrue(np.max(0.5 - delta) > 0.25)

    def test_attack_strength(self):
        """
        If clipping is not done at each iteration (not passing clip_min and
        clip_max to fgm), this attack fails by
        np.mean(orig_labels == new_labels) == .39.
        """
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, ord=np.inf,
                                        clip_min=0.5, clip_max=0.7,
                                        nb_iter=5)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_generate_np_does_not_cache_graph_computation_for_nb_iter(self):

        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, ord=np.inf,
                                        clip_min=-5.0, clip_max=5.0,
                                        nb_iter=10)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

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
        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

        tf.gradients = old_grads

        self.assertTrue(ok[0])


class TestMomentumIterativeMethod(TestBasicIterativeMethod):
    def setUp(self):
        super(TestMomentumIterativeMethod, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = MomentumIterativeMethod(self.model, sess=self.sess)

    def test_generate_np_can_be_called_with_different_decay_factor(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        for dacay_factor in [0.0, 0.5, 1.0]:
            x_adv = self.attack.generate_np(x_val, eps=0.5, ord=np.inf,
                                            dacay_factor=dacay_factor,
                                            clip_min=-5.0, clip_max=5.0)

            delta = np.max(np.abs(x_adv - x_val), axis=1)
            self.assertClose(delta, 0.5)


class TestCarliniWagnerL2(CleverHansTest):
    def setUp(self):
        super(TestCarliniWagnerL2, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = CarliniWagnerL2(self.model, sess=self.sess)

    def test_generate_np_untargeted_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, max_iterations=100,
                                        binary_search_steps=3,
                                        initial_const=1,
                                        clip_min=-5, clip_max=5,
                                        batch_size=10)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_generate_np_targeted_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        feed_labs = np.zeros((100, 2))
        feed_labs[np.arange(100), np.random.randint(0, 1, 100)] = 1
        x_adv = self.attack.generate_np(x_val, max_iterations=100,
                                        binary_search_steps=3,
                                        initial_const=1,
                                        clip_min=-5, clip_max=5,
                                        batch_size=100, y_target=feed_labs)

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(np.argmax(feed_labs, axis=1) == new_labs)
                        > 0.9)

    def test_generate_gives_adversarial_example(self):

        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        feed_labs = np.zeros((100, 2))
        feed_labs[np.arange(100), orig_labs] = 1
        x = tf.placeholder(tf.float32, x_val.shape)
        y = tf.placeholder(tf.float32, feed_labs.shape)

        x_adv_p = self.attack.generate(x, max_iterations=100,
                                       binary_search_steps=3,
                                       initial_const=1,
                                       clip_min=-5, clip_max=5,
                                       batch_size=100, y=y)
        x_adv = self.sess.run(x_adv_p, {x: x_val, y: feed_labs})

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_generate_np_gives_clipped_adversarial_examples(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, max_iterations=10,
                                        binary_search_steps=1,
                                        learning_rate=1e-3,
                                        initial_const=1,
                                        clip_min=-0.2, clip_max=0.3,
                                        batch_size=100)

        self.assertTrue(-0.201 < np.min(x_adv))
        self.assertTrue(np.max(x_adv) < .301)

    def test_generate_np_high_confidence_targeted_examples(self):

        trivial_model = TrivialModel()

        for CONFIDENCE in [0, 2.3]:
            x_val = np.random.rand(10, 1) - .5
            x_val = np.array(x_val, dtype=np.float32)

            feed_labs = np.zeros((10, 2))
            feed_labs[np.arange(10), np.random.randint(0, 2, 10)] = 1
            attack = CarliniWagnerL2(trivial_model, sess=self.sess)
            x_adv = attack.generate_np(x_val,
                                       max_iterations=100,
                                       binary_search_steps=2,
                                       learning_rate=1e-2,
                                       initial_const=1,
                                       clip_min=-10, clip_max=10,
                                       confidence=CONFIDENCE,
                                       y_target=feed_labs,
                                       batch_size=10)

            new_labs = self.sess.run(trivial_model.get_logits(x_adv))

            good_labs = new_labs[np.arange(10), np.argmax(feed_labs, axis=1)]
            bad_labs = new_labs[np.arange(
                10), 1 - np.argmax(feed_labs, axis=1)]

            self.assertClose(CONFIDENCE, np.min(good_labs - bad_labs),
                             atol=1e-1)
            self.assertTrue(np.mean(np.argmax(new_labs, axis=1) ==
                                    np.argmax(feed_labs, axis=1)) > .9)

    def test_generate_np_high_confidence_untargeted_examples(self):

        trivial_model = TrivialModel()

        for CONFIDENCE in [0, 2.3]:
            x_val = np.random.rand(10, 1) - .5
            x_val = np.array(x_val, dtype=np.float32)

            orig_labs = np.argmax(self.sess.run(trivial_model.get_logits(x_val)), axis=1)
            attack = CarliniWagnerL2(trivial_model, sess=self.sess)
            x_adv = attack.generate_np(x_val,
                                       max_iterations=100,
                                       binary_search_steps=2,
                                       learning_rate=1e-2,
                                       initial_const=1,
                                       clip_min=-10, clip_max=10,
                                       confidence=CONFIDENCE,
                                       batch_size=10)

            new_labs = self.sess.run(trivial_model.get_logits(x_adv))

            good_labs = new_labs[np.arange(10), 1 - orig_labs]
            bad_labs = new_labs[np.arange(10), orig_labs]

            self.assertTrue(np.mean(np.argmax(new_labs, axis=1) == orig_labs)
                            == 0)
            self.assertTrue(np.isclose(
                0, np.min(good_labs - (bad_labs + CONFIDENCE)), atol=1e-1))


class TestElasticNetMethod(CleverHansTest):
    def setUp(self):
        super(TestElasticNetMethod, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = ElasticNetMethod(self.model, sess=self.sess)

    def test_generate_np_untargeted_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, max_iterations=100,
                                        binary_search_steps=3,
                                        initial_const=1,
                                        clip_min=-5, clip_max=5,
                                        batch_size=10)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_generate_np_targeted_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        feed_labs = np.zeros((100, 2))
        feed_labs[np.arange(100), np.random.randint(0, 1, 100)] = 1
        x_adv = self.attack.generate_np(x_val, max_iterations=100,
                                        binary_search_steps=3,
                                        initial_const=1,
                                        clip_min=-5, clip_max=5,
                                        batch_size=100, y_target=feed_labs)

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(np.argmax(feed_labs, axis=1) == new_labs) >
                        0.9)

    def test_generate_gives_adversarial_example(self):

        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        feed_labs = np.zeros((100, 2))
        feed_labs[np.arange(100), orig_labs] = 1
        x = tf.placeholder(tf.float32, x_val.shape)
        y = tf.placeholder(tf.float32, feed_labs.shape)

        x_adv_p = self.attack.generate(x, max_iterations=100,
                                       binary_search_steps=3,
                                       initial_const=1,
                                       clip_min=-5, clip_max=5,
                                       batch_size=100, y=y)
        x_adv = self.sess.run(x_adv_p, {x: x_val, y: feed_labs})

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_generate_np_gives_clipped_adversarial_examples(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, max_iterations=10,
                                        binary_search_steps=1,
                                        learning_rate=1e-3,
                                        initial_const=1,
                                        clip_min=-0.2, clip_max=0.3,
                                        batch_size=100)

        self.assertTrue(-0.201 < np.min(x_adv))
        self.assertTrue(np.max(x_adv) < .301)

    def test_generate_np_high_confidence_targeted_examples(self):

        trivial_model = TrivialModel()

        for CONFIDENCE in [0, 2.3]:
            x_val = np.random.rand(10, 1) - .5
            x_val = np.array(x_val, dtype=np.float32)

            feed_labs = np.zeros((10, 2))
            feed_labs[np.arange(10), np.random.randint(0, 2, 10)] = 1
            attack = CarliniWagnerL2(trivial_model, sess=self.sess)
            x_adv = attack.generate_np(x_val,
                                       max_iterations=100,
                                       binary_search_steps=2,
                                       learning_rate=1e-2,
                                       initial_const=1,
                                       clip_min=-10, clip_max=10,
                                       confidence=CONFIDENCE,
                                       y_target=feed_labs,
                                       batch_size=10)

            new_labs = self.sess.run(trivial_model.get_logits(x_adv))

            good_labs = new_labs[np.arange(10), np.argmax(feed_labs, axis=1)]
            bad_labs = new_labs[np.arange(
                10), 1 - np.argmax(feed_labs, axis=1)]

            self.assertTrue(np.isclose(
                0, np.min(good_labs - (bad_labs + CONFIDENCE)), atol=1e-1))
            self.assertTrue(np.mean(np.argmax(new_labs, axis=1) ==
                                    np.argmax(feed_labs, axis=1)) > .9)

    def test_generate_np_high_confidence_untargeted_examples(self):

        trivial_model = TrivialModel()

        for CONFIDENCE in [0, 2.3]:
            x_val = np.random.rand(10, 1) - .5
            x_val = np.array(x_val, dtype=np.float32)

            orig_labs = np.argmax(self.sess.run(trivial_model.get_logits(x_val)), axis=1)
            attack = CarliniWagnerL2(trivial_model, sess=self.sess)
            x_adv = attack.generate_np(x_val,
                                       max_iterations=100,
                                       binary_search_steps=2,
                                       learning_rate=1e-2,
                                       initial_const=1,
                                       clip_min=-10, clip_max=10,
                                       confidence=CONFIDENCE,
                                       batch_size=10)

            new_labs = self.sess.run(trivial_model.get_logits(x_adv))

            good_labs = new_labs[np.arange(10), 1 - orig_labs]
            bad_labs = new_labs[np.arange(10), orig_labs]

            self.assertTrue(np.mean(np.argmax(new_labs, axis=1) == orig_labs)
                            == 0)
            self.assertTrue(np.isclose(
                0, np.min(good_labs - (bad_labs + CONFIDENCE)), atol=1e-1))


class TestSaliencyMapMethod(CleverHansTest):
    def setUp(self):
        super(TestSaliencyMapMethod, self).setUp()

        self.sess = tf.Session()
        self.sess.as_default()
        self.model = DummyModel()
        self.attack = SaliencyMapMethod(self.model, sess=self.sess)

        # initialize model
        with tf.name_scope('dummy_model'):
            self.model(tf.placeholder(tf.float32, shape=(None, 1000)))
        self.sess.run(tf.global_variables_initializer())

        self.attack = SaliencyMapMethod(self.model, sess=self.sess)

    def test_generate_np_targeted_gives_adversarial_example(self):
        x_val = np.random.rand(10, 1000)
        x_val = np.array(x_val, dtype=np.float32)

        feed_labs = np.zeros((10, 10))
        feed_labs[np.arange(10), np.random.randint(0, 9, 10)] = 1
        x_adv = self.attack.generate_np(x_val,
                                        clip_min=-5., clip_max=5.,
                                        y_target=feed_labs)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        worked = np.mean(np.argmax(feed_labs, axis=1) == new_labs)
        self.assertTrue(worked > .9)


class TestDeepFool(CleverHansTest):
    def setUp(self):
        super(TestDeepFool, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = DeepFool(self.model, sess=self.sess)

    def test_generate_np_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, over_shoot=0.02, max_iter=50,
                                        nb_candidate=2, clip_min=-5,
                                        clip_max=5)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_generate_gives_adversarial_example(self):

        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        x = tf.placeholder(tf.float32, x_val.shape)

        x_adv_p = self.attack.generate(x, over_shoot=0.02, max_iter=50,
                                       nb_candidate=2, clip_min=-5, clip_max=5)
        x_adv = self.sess.run(x_adv_p, {x: x_val})

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_generate_np_gives_clipped_adversarial_examples(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, over_shoot=0.02, max_iter=50,
                                        nb_candidate=2, clip_min=-0.2,
                                        clip_max=0.3)

        self.assertTrue(-0.201 < np.min(x_adv))
        self.assertTrue(np.max(x_adv) < .301)


class TestMadryEtAl(CleverHansTest):
    def setUp(self):
        super(TestMadryEtAl, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = MadryEtAl(self.model, sess=self.sess)

    def test_attack_strength(self):
        """
        If clipping is not done at each iteration (not using clip_min and
        clip_max), this attack fails by
        np.mean(orig_labels == new_labels) == .5
        """
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.05,
                                        clip_min=0.5, clip_max=0.7,
                                        nb_iter=5)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_clip_eta(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.1,
                                        nb_iter=5)

        delta = np.max(np.abs(x_adv - x_val), axis=1)
        self.assertTrue(np.all(delta <= 1.))

    def test_generate_np_gives_clipped_adversarial_examples(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.1,
                                        nb_iter=5,
                                        clip_min=-0.2, clip_max=0.3)

        self.assertTrue(-0.201 < np.min(x_adv))
        self.assertTrue(np.max(x_adv) < .301)

    def test_multiple_initial_random_step(self):
        """
        This test generates multiple adversarial examples until an adversarial
        example is generated with a different label compared to the original
        label. This is the procedure suggested in Madry et al. (2017).

        This test will fail if an initial random step is not taken (error>0.5).
        """
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs_multi = orig_labs.copy()

        # Generate multiple adversarial examples
        for i in range(10):
            x_adv = self.attack.generate_np(x_val, eps=.5, eps_iter=0.05,
                                            clip_min=0.5, clip_max=0.7,
                                            nb_iter=2)
            new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

            # Examples for which we have not found adversarial examples
            I = (orig_labs == new_labs_multi)
            new_labs_multi[I] = new_labs[I]

        self.assertTrue(np.mean(orig_labs == new_labs_multi) < 0.1)


class TestFastFeatureAdversaries(CleverHansTest):
    def setUp(self):
        super(TestFastFeatureAdversaries, self).setUp()

        def make_imagenet_cnn(input_shape=(None, 224, 224, 3)):
            """
            Similar CNN to AlexNet.
            """
            import cleverhans_tutorials.tutorial_models as t_models
            layers = [t_models.Conv2D(96, (3, 3), (2, 2), "VALID"),
                      t_models.ReLU(),
                      t_models.Conv2D(256, (3, 3), (2, 2), "VALID"),
                      t_models.ReLU(),
                      t_models.Conv2D(384, (3, 3), (2, 2), "VALID"),
                      t_models.ReLU(),
                      t_models.Conv2D(384, (3, 3), (2, 2), "VALID"),
                      t_models.ReLU(),
                      t_models.Conv2D(256, (3, 3), (2, 2), "VALID"),
                      t_models.ReLU(),
                      t_models.Flatten(),
                      t_models.Linear(4096),
                      t_models.ReLU(),
                      t_models.Linear(4096),
                      t_models.ReLU(),
                      t_models.Linear(1000),
                      t_models.Softmax()]
            layers[-3].name = 'fc7'

            model = t_models.MLP(layers, input_shape)
            return model

        self.input_shape = [10, 224, 224, 3]
        self.sess = tf.Session()
        self.model = make_imagenet_cnn(self.input_shape)
        self.attack = FastFeatureAdversaries(self.model)

    def test_attack_strength(self):
        """
        This test generates a random source and guide and feeds them in a
        randomly initialized CNN. Checks if an adversarial example can get
        at least 50% closer to the guide compared to the original distance of
        the source and the guide.
        """
        tf.set_random_seed(1234)
        input_shape = self.input_shape
        x_src = tf.abs(tf.random_uniform(input_shape, 0., 1.))
        x_guide = tf.abs(tf.random_uniform(input_shape, 0., 1.))

        layer = 'fc7'
        attack_params = {'eps': 5./256, 'clip_min': 0., 'clip_max': 1.,
                         'nb_iter': 10, 'eps_iter': 0.005,
                         'layer': layer}
        x_adv = self.attack.generate(x_src, x_guide, **attack_params)
        h_adv = self.model.fprop(x_adv)[layer]
        h_src = self.model.fprop(x_src)[layer]
        h_guide = self.model.fprop(x_guide)[layer]

        init = tf.global_variables_initializer()
        self.sess.run(init)

        ha, hs, hg, xa, xs, xg = self.sess.run(
            [h_adv, h_src, h_guide, x_adv, x_src, x_guide])
        d_as = np.sqrt(((hs-ha)*(hs-ha)).sum())
        d_ag = np.sqrt(((hg-ha)*(hg-ha)).sum())
        d_sg = np.sqrt(((hg-hs)*(hg-hs)).sum())
        print("L2 distance between source and adversarial example `%s`: %.4f" %
              (layer, d_as))
        print("L2 distance between guide and adversarial example `%s`: %.4f" %
              (layer, d_ag))
        print("L2 distance between source and guide `%s`: %.4f" %
              (layer, d_sg))
        print("d_ag/d_sg*100 `%s`: %.4f" % (layer, d_ag*100/d_sg))
        self.assertTrue(d_ag*100/d_sg < 50.)


class TestLBFGS(CleverHansTest):
    def setUp(self):
        super(TestLBFGS, self).setUp()

        self.sess = tf.Session()
        self.model = SimpleModel()
        self.attack = LBFGS(self.model, sess=self.sess)

    def test_generate_np_targeted_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        feed_labs = np.zeros((100, 2))
        feed_labs[np.arange(100), np.random.randint(0, 1, 100)] = 1
        x_adv = self.attack.generate_np(x_val, max_iterations=100,
                                        binary_search_steps=3,
                                        initial_const=1,
                                        clip_min=-5, clip_max=5,
                                        batch_size=100, y_target=feed_labs)

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(np.argmax(feed_labs, axis=1) == new_labs)
                        > 0.9)

    def test_generate_targeted_gives_adversarial_example(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        feed_labs = np.zeros((100, 2))
        feed_labs[np.arange(100), np.random.randint(0, 1, 100)] = 1
        x = tf.placeholder(tf.float32, x_val.shape)
        y = tf.placeholder(tf.float32, feed_labs.shape)

        x_adv_p = self.attack.generate(x, max_iterations=100,
                                       binary_search_steps=3,
                                       initial_const=1,
                                       clip_min=-5, clip_max=5,
                                       batch_size=100, y_target=y)
        x_adv = self.sess.run(x_adv_p, {x: x_val, y: feed_labs})

        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        self.assertTrue(np.mean(np.argmax(feed_labs, axis=1) == new_labs)
                        > 0.9)

    def test_generate_np_gives_clipped_adversarial_examples(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        feed_labs = np.zeros((100, 2))
        feed_labs[np.arange(100), np.random.randint(0, 1, 100)] = 1
        x_adv = self.attack.generate_np(x_val, max_iterations=10,
                                        binary_search_steps=1,
                                        initial_const=1,
                                        clip_min=-0.2, clip_max=0.3,
                                        batch_size=100, y_target=feed_labs)

        self.assertTrue(-0.201 < np.min(x_adv))
        self.assertTrue(np.max(x_adv) < .301)


if __name__ == '__main__':
    unittest.main()
