# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import unittest

import numpy as np
from nose.plugins.skip import SkipTest
import tensorflow as tf
# pylint bug on next line
import tensorflow.contrib.slim as slim  # pylint: disable=no-name-in-module

from cleverhans.devtools.checks import CleverHansTest
from cleverhans import attacks
from cleverhans.attacks import Attack, SPSA
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import VirtualAdversarialMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import FastFeatureAdversaries
from cleverhans.attacks import LBFGS
from cleverhans.attacks import SpatialTransformationMethod
from cleverhans.attacks import HopSkipJumpAttack
from cleverhans.attacks import SparseL1Descent
from cleverhans.initializers import HeReLuNormalInitializer
from cleverhans.model import Model


class SimpleModel(Model):
  """
  A very simple neural network
  """

  def __init__(self, scope='simple', nb_classes=2, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())

  def fprop(self, x, **kwargs):
    del kwargs
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      w1 = tf.constant([[1.5, .3], [-2, 0.3]],
                       dtype=tf.as_dtype(x.dtype))
      w2 = tf.constant([[-2.4, 1.2], [0.5, -2.3]],
                       dtype=tf.as_dtype(x.dtype))
    h1 = tf.nn.sigmoid(tf.matmul(x, w1))
    res = tf.matmul(h1, w2)
    return {self.O_LOGITS: res,
            self.O_PROBS: tf.nn.softmax(res)}


class TrivialModel(Model):
  """
  A linear model with two weights
  """

  def __init__(self, scope='trivial', nb_classes=2, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())

  def fprop(self, x, **kwargs):
    del kwargs
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      w1 = tf.constant([[1, -1]], dtype=tf.float32)
    res = tf.matmul(x, w1)
    return {self.O_LOGITS: res,
            self.O_PROBS: tf.nn.softmax(res)}


class DummyModel(Model):
  """
  A simple model based on slim
  """

  def __init__(self, scope='dummy_model', nb_classes=10, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())

  def fprop(self, x, **kwargs):
    del kwargs
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      net = slim.flatten(x)
      net = slim.fully_connected(net, 60)
      logits = slim.fully_connected(net, 10, activation_fn=None)
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits)}


class TestAttackClassInitArguments(CleverHansTest):

  def test_model(self):
    sess = tf.Session()

    # Exception is thrown when model does not have __call__ attribute
    with self.assertRaises(Exception) as context:
      model = tf.placeholder(tf.float32, shape=(None, 10))
      Attack(model, sess=sess)
    self.assertTrue(context.exception)

  def test_sess(self):
    # Test that it is permitted to provide no session.
    # The session still needs to be created prior to running the attack.
    # TODO: does anyone know why we need to make an unused session and put it in a with statement?
    with tf.Session():
      Attack(Model('model', 10, {}), sess=None)

  def test_sess_generate_np(self):
    model = Model('model', 10, {})

    class DummyAttack(Attack):
      def generate(self, x, **kwargs):
        return x

    # Test that generate_np is NOT permitted without a session.
    # The session still needs to be created prior to running the attack.
    # TODO: does anyone know why we need to make an unused session and put it in a with statement?
    with tf.Session():
      attack = DummyAttack(model, sess=None)
      with self.assertRaises(Exception) as context:
        attack.generate_np(0.)
      self.assertTrue(context.exception)


class TestParseParams(CleverHansTest):
  def test_parse(self):
    sess = tf.Session()

    test_attack = Attack(Model('model', 10, {}), sess=sess)
    self.assertTrue(test_attack.parse_params({}))


class TestVirtualAdversarialMethod(CleverHansTest):
  def setUp(self):
    super(TestVirtualAdversarialMethod, self).setUp()

    self.sess = tf.Session()
    self.sess.as_default()
    self.model = DummyModel('virtual_adv_dummy_model')
    self.attack = VirtualAdversarialMethod(self.model, sess=self.sess)

    # initialize model
    with tf.name_scope('virtual_adv_dummy_model'):
      self.model.get_probs(tf.placeholder(tf.float32, shape=(None, 1000)))
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
    perturbation_norm = np.sqrt(np.sum(perturbation ** 2, axis=1))
    # test perturbation norm
    self.assertClose(perturbation_norm, self.attack.eps)


class CommonAttackProperties(CleverHansTest):
  """
  Abstract base class shared by the tests for many attacks that want
  to check the same properties.
  """

  def setUp(self):
    # Inheritance doesn't really work with tests.
    # nosetests always wants to run this class because it is a
    # CleverHansTest subclass, but this class is meant to just
    # be abstract.
    # Before this class was the tests for FastGradientMethod but
    # people kept inheriting from it for other attacks so it was
    # impossible to write tests specifically for FastGradientMethod.
    # pylint: disable=unidiomatic-typecheck
    if type(self) is CommonAttackProperties:
      raise SkipTest()

    super(CommonAttackProperties, self).setUp()
    self.sess = tf.Session()
    self.model = SimpleModel()

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
      delta = np.sum(np.square(x_adv - x_val), axis=1) ** .5

    return x_val, x_adv, delta

  def help_generate_np_gives_adversarial_example(self, ord, eps=.5,
                                                 **kwargs):
    x_val, x_adv, delta = self.generate_adversarial_examples_np(ord, eps,
                                                                **kwargs)
    self.assertLess(np.max(np.abs(delta-eps)), 1e-3)
    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.max(np.mean(orig_labs == new_labs)), .5)

  def test_invalid_input(self):
    x_val = -np.ones((2, 2), dtype='float32')
    with self.assertRaises(tf.errors.InvalidArgumentError) as context:
      self.attack.generate_np(x_val, eps=1., clip_min=0., clip_max=1.)
    self.assertTrue(context.exception)

  def test_generate_np_gives_adversarial_example_linfinity(self):
    self.help_generate_np_gives_adversarial_example(np.infty)

  def test_generate_np_gives_adversarial_example_l1(self):
    self.help_generate_np_gives_adversarial_example(1)

  def test_generate_np_gives_adversarial_example_l2(self):
    self.help_generate_np_gives_adversarial_example(2)

  def test_generate_respects_dtype(self):
    self.attack = FastGradientMethod(self.model, sess=self.sess,
                                     dtypestr='float64')
    x = tf.placeholder(dtype=tf.float64, shape=(100, 2))
    x_adv = self.attack.generate(x)
    self.assertEqual(x_adv.dtype, tf.float64)

  def test_targeted_generate_np_gives_adversarial_example(self):
    random_labs = np.random.random_integers(0, 1, 100)
    random_labs_one_hot = np.zeros((100, 2))
    random_labs_one_hot[np.arange(100), random_labs] = 1

    try:
      _, x_adv, delta = self.generate_adversarial_examples_np(
          eps=.5, ord=np.inf, y_target=random_labs_one_hot)
    except NotImplementedError:
      raise SkipTest()

    self.assertLessEqual(np.max(delta), 0.5001)

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertTrue(np.mean(random_labs == new_labs) > 0.7)

  def test_generate_np_can_be_called_with_different_eps(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    for eps in [0.1, 0.2, 0.3, 0.4]:
      x_adv = self.attack.generate_np(x_val, eps=eps, ord=np.inf,
                                      clip_min=-5.0, clip_max=5.0)

      delta = np.max(np.abs(x_adv - x_val), axis=1)
      self.assertLessEqual(np.max(delta), eps+1e-4)

  def test_generate_can_be_called_with_different_eps(self):
    # It is crtical that this test uses generate and not generate_np.
    # All the other tests use generate_np. Even though generate_np calls
    # generate, it does so in a very standardized way, e.g. with eps
    # always converted to a tensorflow placeholder, so the other tests
    # based on generate_np do not exercise the generate API very well.
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)
    x = tf.placeholder(tf.float32, x_val.shape)

    for eps in [0.1, 0.2, 0.3, 0.4]:
      x_adv = self.attack.generate(x, eps=eps, ord=np.inf,
                                   clip_min=-5.0, clip_max=5.0)
      x_adv = self.sess.run(x_adv, feed_dict={x: x_val})

      delta = np.max(np.abs(x_adv - x_val), axis=1)
      self.assertLessEqual(np.max(delta), eps + 1e-4)

  def test_generate_np_clip_works_as_expected(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, eps=0.5, ord=np.inf,
                                    clip_min=-0.2, clip_max=0.1,
                                    sanity_checks=False)

    self.assertClose(np.min(x_adv), -0.2)
    self.assertClose(np.max(x_adv), 0.1)


class TestFastGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestFastGradientMethod, self).setUp()

    self.attack = FastGradientMethod(self.model, sess=self.sess)


class TestOptimizeLinear(CleverHansTest):
  """
  Tests for the `optimize_linear` function
  """

  def setUp(self):
    super(TestOptimizeLinear, self).setUp()
    self.sess = tf.Session()
    self.model = SimpleModel()

  def test_optimize_linear_linf(self):

    grad = tf.placeholder(tf.float32, shape=[1, 2])

    # Build the graph for the attack
    eta = attacks.optimize_linear(grad, eps=1., ord=np.inf)
    objective = tf.reduce_sum(grad * eta)

    grad_val = np.array([[1., -2.]])
    eta, objective = self.sess.run([eta, objective],
                                   feed_dict={grad: grad_val})

    # Make sure the objective is optimal.
    # This is the solution obtained by doing the algebra by hand.
    self.assertClose(objective, np.abs(grad_val).sum())
    # Make sure the constraint is respected.
    # Also, for a linear function, the constraint will always be tight.
    self.assertClose(np.abs(eta), 1.)

  def test_optimize_linear_l2(self):

    grad = tf.placeholder(tf.float32, shape=[1, 2])

    # Build the graph for the attack
    eta = attacks.optimize_linear(grad, eps=1., ord=2)
    objective = tf.reduce_sum(grad * eta)

    grad_val = np.array([[np.sqrt(.5), -np.sqrt(.5)]])
    eta, objective = self.sess.run([eta, objective],
                                   feed_dict={grad: grad_val})

    # Make sure the objective is optimal.
    # This is the solution obtained by doing the algebra by hand.
    self.assertClose(objective, 1.)
    # Make sure the constraint is respected.
    # Also, for a linear function, the constraint will always be tight.
    self.assertClose(np.sqrt(np.square(eta).sum()), 1.)

  def test_optimize_linear_l1(self):

    # This test makes sure that `optimize_linear` actually finds the optimal
    # perturbation for ord=1.
    # A common misconcpetion is that FGM for ord=1 consists of dividing
    # the gradient by its L1 norm.
    # If you do that for the problem in this unit test, you'll get an
    # objective function value of ~1.667. The optimal result is 2.

    # We need just one example in the batch and two features to show the
    # common misconception is suboptimal.
    grad = tf.placeholder(tf.float32, shape=[1, 2])

    # Build the graph for the attack
    eta = attacks.optimize_linear(grad, eps=1., ord=1)
    objective = tf.reduce_sum(grad * eta)

    # Make sure the largest entry of the gradient for the test case is
    # negative, to catch
    # the potential bug where we forget to handle the sign of the gradient
    eta, objective = self.sess.run([eta, objective],
                                   feed_dict={grad: np.array([[1., -2.]])})

    # Make sure the objective is optimal.
    # This is the solution obtained by doing the algebra by hand.
    self.assertClose(objective, 2.)
    # Make sure the constraint is respected.
    # Also, for a linear function, the constraint will always be tight.
    self.assertClose(np.abs(eta).sum(), 1.)

  def test_optimize_linear_l1_ties(self):

    # This test makes sure that `optimize_linear` handles ties in gradient
    # magnitude correctly when ord=1.

    # We need just one example in the batch and two features to construct
    # a tie.
    grad = tf.placeholder(tf.float32, shape=[1, 2])

    # Build the graph for the attack
    eta = attacks.optimize_linear(grad, eps=1., ord=1)
    objective = tf.reduce_sum(grad * eta)

    # Run a test case with a tie for largest absolute value.
    # Make one feature negative to make sure we're checking for ties in
    # absolute value, not raw value.
    eta, objective = self.sess.run([eta, objective],
                                   feed_dict={grad: np.array([[2., -2.]])})

    # Make sure the objective is optimal.
    # This is the solution obtained by doing the algebra by hand.
    self.assertClose(objective, 2.)
    # Make sure the constraint is respected.
    # Also, for a linear function, the constraint will always be tight.
    self.assertClose(np.abs(eta).sum(), 1.)


class TestSPSA(CleverHansTest):
  def setUp(self):
    super(TestSPSA, self).setUp()

    self.sess = tf.Session()
    self.model = SimpleModel()
    self.attack = SPSA(self.model, sess=self.sess)

  def test_attack_strength(self):
    n_samples = 10
    x_val = np.random.rand(n_samples, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # The SPSA attack currently uses non-one-hot labels
    # TODO: change this to use standard cleverhans label conventions
    feed_labs = np.random.randint(0, 2, n_samples)

    x_input = tf.placeholder(tf.float32, shape=(1, 2))
    y_label = tf.placeholder(tf.int32, shape=(1,))

    x_adv_op = self.attack.generate(
        x_input, y=y_label,
        epsilon=.5, num_steps=100, batch_size=64, spsa_iters=1,
        clip_min=0., clip_max=1.
    )

    all_x_adv = []
    for i in range(n_samples):
      x_adv_np = self.sess.run(x_adv_op, feed_dict={
          x_input: np.expand_dims(x_val[i], axis=0),
          y_label: np.expand_dims(feed_labs[i], axis=0),
      })
      all_x_adv.append(x_adv_np[0])

    x_adv = np.vstack(all_x_adv)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertTrue(np.mean(feed_labs == new_labs) < 0.1)

  def test_attack_strength_np(self):
    # Same test as test_attack_strength, but uses generate_np interface
    n_samples = 10
    x_val = np.random.rand(n_samples, 2)
    x_val = np.array(x_val, dtype=np.float32)

    feed_labs = np.random.randint(0, 2, n_samples, dtype='int32')

    all_x_adv = []
    for i in range(n_samples):
      x_adv_np = self.attack.generate_np(
          np.expand_dims(x_val[i], axis=0),
          y=np.expand_dims(feed_labs[i], axis=0),
          eps=.5, nb_iter=100, spsa_samples=64, spsa_iters=1,
          clip_min=0., clip_max=1.
      )
      all_x_adv.append(x_adv_np[0])

    x_adv = np.vstack(all_x_adv)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.mean(feed_labs == new_labs), 0.1)

  def test_attack_strength_np_batched(self):
    # Same test as test_attack_strength_np, but batched
    n_samples = 10
    x_val = np.random.rand(n_samples, 2)
    x_val = np.array(x_val, dtype=np.float32)

    feed_labs = np.random.randint(0, 2, n_samples, dtype='int32')
    x_adv = self.attack.generate_np(
        x_val, y=feed_labs, eps=.5, nb_iter=100, spsa_samples=64,
        spsa_iters=1, clip_min=0., clip_max=1.)

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.mean(feed_labs == new_labs), 0.1)

  def test_label_argument_int64(self):
    x_val = np.random.rand(1, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # Try calling alternating with int32 and int64 and ensure it works
    for dtype in [np.int32, np.int64, np.int32, np.int64]:
      self.attack.generate_np(x_val, y=np.zeros(1, dtype=dtype), eps=.5,
                              nb_iter=5, spsa_samples=64,
                              spsa_iters=1, clip_min=0., clip_max=1.)


class TestProjectedGradientDescent(CommonAttackProperties):
  def setUp(self):
    super(TestProjectedGradientDescent, self).setUp()

    self.sess = tf.Session()
    self.model = SimpleModel()
    self.attack = ProjectedGradientDescent(self.model, sess=self.sess)

  def test_generate_np_gives_adversarial_example_linfinity(self):
    self.help_generate_np_gives_adversarial_example(ord=np.infty, eps=.5,
                                                    nb_iter=20)

  def test_generate_np_gives_adversarial_example_l1(self):
    try:
      self.help_generate_np_gives_adversarial_example(ord=1, eps=.5,
                                                      nb_iter=20)
    except NotImplementedError:
      raise SkipTest()

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
      try:
        _, _, delta = self.generate_adversarial_examples_np(
            ord=ord, eps=.5, nb_iter=10, eps_iter=.01)
      except NotImplementedError:
        # Don't raise SkipTest because it will skip the rest of the for loop
        continue
      self.assertTrue(np.max(0.5 - delta) > 0.25)

  def test_attack_strength_linf(self):
    """
    If clipping is not done at each iteration (not passing clip_min and
    clip_max to fgm), this attack fails by
    np.mean(orig_labels == new_labels) == .39.
    """
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # sanity checks turned off because this test initializes outside
    # the valid range.
    x_adv = self.attack.generate_np(x_val, eps=1.0, ord=np.inf,
                                    clip_min=0.5, clip_max=0.7,
                                    nb_iter=5, sanity_checks=False)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.mean(orig_labs == new_labs), 0.1)

  def test_attack_strength_l2(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # sanity checks turned off because this test initializes outside
    # the valid range.
    x_adv = self.attack.generate_np(x_val, eps=1, ord=2,
                                    clip_min=0.5, clip_max=0.7,
                                    nb_iter=5, sanity_checks=False)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.mean(orig_labs == new_labs), 0.1)

  def test_grad_clip_l2(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # sanity checks turned off because this test initializes outside
    # the valid range.
    x_adv = self.attack.generate_np(x_val, eps=1, ord=2,
                                    clip_min=0.5, clip_max=0.7, clip_grad=True,
                                    nb_iter=10, sanity_checks=False)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.mean(orig_labs == new_labs), 0.1)

  def test_clip_eta_linf(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, ord=np.inf, eps=1.0, eps_iter=0.1,
                                    nb_iter=5)

    delta = np.max(np.abs(x_adv - x_val), axis=1)
    self.assertLessEqual(np.max(delta), 1.)

  def test_clip_eta_l2(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, ord=2, eps=1.0, eps_iter=0.1,
                                    nb_iter=5)

    delta = np.sqrt(np.sum(np.square(x_adv - x_val), axis=1))
    # this projection is less numerically stable so give it some slack
    self.assertLessEqual(np.max(delta), 1. + 1e-6)

  def test_generate_np_gives_clipped_adversarial_examples(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    for ord in [1, 2, np.infty]:
      try:
        x_adv = self.attack.generate_np(x_val, ord=ord, eps=1.0, eps_iter=0.1,
                                        nb_iter=5,
                                        clip_min=-0.2, clip_max=0.3,
                                        sanity_checks=False)

        self.assertLess(-0.201, np.min(x_adv))
        self.assertLess(np.max(x_adv), .301)
      except NotImplementedError:
        # Don't raise SkipTest because it will skip the rest of the for loop
        continue

  def test_generate_np_does_not_cache_graph_computation_for_nb_iter(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # Call it once
    x_adv = self.attack.generate_np(x_val, eps=1.0, ord=np.inf,
                                    clip_min=-5.0, clip_max=5.0,
                                    nb_iter=10)

    # original labels
    np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    # new labels
    np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    # Call it again
    ok = [False]
    old_grads = tf.gradients
    try:
      def fn(*x, **y):
        ok[0] = True
        return old_grads(*x, **y)

      tf.gradients = fn

      x_adv = self.attack.generate_np(x_val, eps=1.0, ord=np.inf,
                                      clip_min=-5.0, clip_max=5.0,
                                      nb_iter=11)
    finally:
      tf.gradients = old_grads

    # original labels
    np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    # new labels
    np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(ok[0])

  def test_multiple_initial_random_step(self):
    """
    This test generates multiple adversarial examples until an adversarial
    example is generated with a different label compared to the original
    label. This is the procedure suggested in Madry et al. (2017).

    This test will fail if an initial random step is not taken (error>0.5).
    """
    x_val = np.array(np.random.rand(100, 2), dtype=np.float32)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs_multi = orig_labs.copy()

    # Generate multiple adversarial examples
    for _ in range(10):
      x_adv = self.attack.generate_np(x_val, eps=.5, eps_iter=0.05,
                                      clip_min=0.5, clip_max=0.7,
                                      nb_iter=2, sanity_checks=False)
      new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

      # Examples for which we have not found adversarial examples
      I = (orig_labs == new_labs_multi)
      new_labs_multi[I] = new_labs[I]

    self.assertLess(np.mean(orig_labs == new_labs_multi), 0.5)


class TestSparseL1Descent(CleverHansTest):
  def setUp(self):
    super(TestSparseL1Descent, self).setUp()

    self.sess = tf.Session()
    self.model = SimpleModel()
    self.attack = SparseL1Descent(self.model, sess=self.sess)

  def generate_adversarial_examples_np(self, eps, **kwargs):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, eps=eps,
                                    clip_min=-5, clip_max=5, **kwargs)
    delta = np.sum(np.abs(x_adv - x_val), axis=1)

    return x_val, x_adv, delta

  def help_generate_np_gives_adversarial_example(self, eps=2.0, **kwargs):
    x_val, x_adv, delta = self.generate_adversarial_examples_np(eps, **kwargs)
    self.assertLess(np.max(np.abs(delta-eps)), 1e-3)
    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.max(np.mean(orig_labs == new_labs)), .5)

  def test_invalid_input(self):
    x_val = -np.ones((2, 2), dtype='float32')
    with self.assertRaises(tf.errors.InvalidArgumentError) as context:
      self.attack.generate_np(x_val, eps=10.0, clip_min=0., clip_max=1.)
    self.assertTrue(context.exception)

  def test_generate_np_gives_adversarial_example(self):
    self.help_generate_np_gives_adversarial_example()

  def test_targeted_generate_np_gives_adversarial_example(self):
    random_labs = np.random.random_integers(0, 1, 100)
    random_labs_one_hot = np.zeros((100, 2))
    random_labs_one_hot[np.arange(100), random_labs] = 1

    _, x_adv, delta = self.generate_adversarial_examples_np(
        eps=10, y_target=random_labs_one_hot)

    self.assertLessEqual(np.max(delta), 10.001)

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertTrue(np.mean(random_labs == new_labs) > 0.7)

  def test_generate_np_can_be_called_with_different_eps(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    for eps in [10, 20, 30, 40]:
      x_adv = self.attack.generate_np(x_val, eps=eps,
                                      clip_min=-5.0, clip_max=5.0)

      delta = np.max(np.abs(x_adv - x_val), axis=1)
      self.assertLessEqual(np.max(delta), eps+1e-4)

  def test_generate_can_be_called_with_different_eps(self):
    # It is crtical that this test uses generate and not generate_np.
    # All the other tests use generate_np. Even though generate_np calls
    # generate, it does so in a very standardized way, e.g. with eps
    # always converted to a tensorflow placeholder, so the other tests
    # based on generate_np do not exercise the generate API very well.
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)
    x = tf.placeholder(tf.float32, x_val.shape)

    for eps in [10, 20, 30, 40]:
      x_adv = self.attack.generate(x, eps=eps, clip_min=-5.0, clip_max=5.0)
      x_adv = self.sess.run(x_adv, feed_dict={x: x_val})

      delta = np.max(np.abs(x_adv - x_val), axis=1)
      self.assertLessEqual(np.max(delta), eps + 1e-4)

  def test_generate_np_clip_works_as_expected(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, eps=10, nb_iter=20, rand_init=True,
                                    clip_min=-0.2, clip_max=0.1,
                                    sanity_checks=False)

    self.assertClose(np.min(x_adv), -0.2)
    self.assertClose(np.max(x_adv), 0.1)

  def test_do_not_reach_lp_boundary(self):
    """
    Make sure that iterative attack don't reach boundary of Lp
    neighbourhood if nb_iter * eps_iter is relatively small compared to
    epsilon.
    """

    _, _, delta = self.generate_adversarial_examples_np(
        eps=.5, nb_iter=10, eps_iter=.01)

    self.assertTrue(np.max(0.5 - delta) > 0.25)

  def test_generate_np_gives_clipped_adversarial_examples(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.1,
                                    nb_iter=5,
                                    clip_min=-0.2, clip_max=0.3,
                                    sanity_checks=False)

    self.assertLess(-0.201, np.min(x_adv))
    self.assertLess(np.max(x_adv), .301)

  def test_generate_np_does_not_cache_graph_computation_for_nb_iter(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # Call it once
    x_adv = self.attack.generate_np(x_val, eps=1.0,
                                    clip_min=-5.0, clip_max=5.0,
                                    nb_iter=10)

    # original labels
    np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    # new labels
    np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    # Call it again
    ok = [False]
    old_grads = tf.gradients
    try:
      def fn(*x, **y):
        ok[0] = True
        return old_grads(*x, **y)

      tf.gradients = fn

      x_adv = self.attack.generate_np(x_val, eps=1.0,
                                      clip_min=-5.0, clip_max=5.0,
                                      nb_iter=11)
    finally:
      tf.gradients = old_grads

    # original labels
    np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    # new labels
    np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(ok[0])

  def test_clip_eta(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.1, nb_iter=5)

    delta = np.sum(np.abs(x_adv - x_val), axis=1)
    # this projection is less numerically stable so give it some slack
    self.assertLessEqual(np.max(delta), 1. + 1e-6)

  def test_attack_strength(self):
    """
    Without clipped gradients, we achieve
    np.mean(orig_labels == new_labels) == 0.31.
    """
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # sanity checks turned off because this test initializes outside
    # the valid range.
    x_adv = self.attack.generate_np(x_val, eps=10.0, rand_init=True,
                                    clip_min=0.5, clip_max=0.7,
                                    nb_iter=10, sanity_checks=False)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.mean(orig_labs == new_labs), 0.4)
    self.assertGreater(np.mean(orig_labs == new_labs), 0.2)

  def test_grad_clip(self):
    """
    With clipped gradients, we achieve
    np.mean(orig_labels == new_labels) == 0.0
    """
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # sanity checks turned off because this test initializes outside
    # the valid range.
    x_adv = self.attack.generate_np(x_val, eps=10.0, rand_init=True,
                                    clip_min=0.5, clip_max=0.7,
                                    clip_grad=True,
                                    nb_iter=10, sanity_checks=False)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertLess(np.mean(orig_labs == new_labs), 0.1)

  def test_sparsity(self):

    # use a model with larger input dimensionality for this test.
    self.model = DummyModel('sparse_l1_descent_dummy_model')
    self.attack = SparseL1Descent(self.model, sess=self.sess)

    # initialize model
    with tf.name_scope('sparse_l1_descent_dummy_model'):
      self.model.get_probs(tf.placeholder(tf.float32, shape=(None, 1000)))
    self.sess.run(tf.global_variables_initializer())

    x_val = np.random.rand(100, 1000)
    x_val = np.array(x_val, dtype=np.float32)

    for q in [1, 9, 25.8, 50, 75.4, 90.2, 99, 99.9]:
      x_adv = self.attack.generate_np(x_val, eps=5.0, grad_sparsity=q,
                                      nb_iter=1, sanity_checks=False)

      numzero = np.sum(x_adv - x_val == 0, axis=-1)
      self.assertAlmostEqual(q * 1000.0 / 100.0, np.mean(numzero), delta=1)

  def test_grad_sparsity_checks(self):
    # test that the attacks allows `grad_sparsity` to be specified as a scalar
    # in (0, 100) or as a vector.

    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    # scalar values out of range
    with self.assertRaises(ValueError) as context:
      self.attack.generate(x_val, sanity_checks=False, grad_sparsity=0)
    self.assertTrue(context.exception)

    with self.assertRaises(ValueError) as context:
      self.attack.generate(x_val, sanity_checks=False, grad_sparsity=100)
    self.assertTrue(context.exception)

    # sparsity as 2D array should fail
    with self.assertRaises(ValueError) as context:
      gs = tf.random.uniform(shape=(100, 2), minval=90, maxval=99)
      self.attack.generate(x_val, sanity_checks=False, grad_sparsity=gs)
    self.assertTrue(context.exception)

    # sparsity as 1D array should succeed
    gs = tf.random.uniform(shape=(100,), minval=90, maxval=99)
    x_adv = self.attack.generate(x_val, sanity_checks=False, grad_sparsity=gs)
    self.assertTrue(np.array_equal(x_adv.get_shape().as_list(), [100, 2]))

    # sparsity vector of wrong size should fail
    with self.assertRaises(ValueError) as context:
      gs = tf.random.uniform(shape=(101,), minval=90, maxval=99)
      x_adv = self.attack.generate(x_val, sanity_checks=False, grad_sparsity=gs)
    self.assertTrue(context.exception)


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

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

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

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(np.mean(np.argmax(feed_labs, axis=1) == new_labs)
                    > 0.9)

  def test_generate_gives_adversarial_example(self):

    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    feed_labs = np.zeros((100, 2))
    feed_labs[np.arange(100), orig_labs] = 1
    x = tf.placeholder(tf.float32, x_val.shape)
    y = tf.placeholder(tf.float32, feed_labs.shape)

    x_adv_p = self.attack.generate(x, max_iterations=100,
                                   binary_search_steps=3,
                                   initial_const=1,
                                   clip_min=-5, clip_max=5,
                                   batch_size=100, y=y)
    self.assertEqual(x_val.shape, x_adv_p.shape)
    x_adv = self.sess.run(x_adv_p, {x: x_val, y: feed_labs})

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

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

      orig_labs = np.argmax(
          self.sess.run(trivial_model.get_logits(x_val)), axis=1)
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

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

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

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(np.mean(np.argmax(feed_labs, axis=1) == new_labs) >
                    0.9)

  def test_generate_gives_adversarial_example(self):

    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    feed_labs = np.zeros((100, 2))
    feed_labs[np.arange(100), orig_labs] = 1
    x = tf.placeholder(tf.float32, x_val.shape)
    y = tf.placeholder(tf.float32, feed_labs.shape)

    x_adv_p = self.attack.generate(x, max_iterations=100,
                                   binary_search_steps=3,
                                   initial_const=1,
                                   clip_min=-5, clip_max=5,
                                   batch_size=100, y=y)
    self.assertEqual(x_val.shape, x_adv_p.shape)
    x_adv = self.sess.run(x_adv_p, {x: x_val, y: feed_labs})

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

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

      orig_labs = np.argmax(
          self.sess.run(trivial_model.get_logits(x_val)), axis=1)
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
      self.model.get_logits(tf.placeholder(tf.float32, shape=(None, 1000)))
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
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

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

    x_adv = self.attack.generate_np(x_val, overshoot=0.02, max_iter=50,
                                    nb_candidate=2, clip_min=-5,
                                    clip_max=5)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

  def test_generate_gives_adversarial_example(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    x = tf.placeholder(tf.float32, x_val.shape)

    x_adv_p = self.attack.generate(x, overshoot=0.02, max_iter=50,
                                   nb_candidate=2, clip_min=-5, clip_max=5)
    self.assertEqual(x_val.shape, x_adv_p.shape)
    x_adv = self.sess.run(x_adv_p, {x: x_val})

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

  def test_generate_np_gives_clipped_adversarial_examples(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    x_adv = self.attack.generate_np(x_val, overshoot=0.02, max_iter=50,
                                    nb_candidate=2, clip_min=-0.2,
                                    clip_max=0.3)

    self.assertTrue(-0.201 < np.min(x_adv))
    self.assertTrue(np.max(x_adv) < .301)


class TestMomentumIterativeMethod(TestProjectedGradientDescent):
  def setUp(self):
    super(TestMomentumIterativeMethod, self).setUp()

    self.attack = MomentumIterativeMethod(self.model, sess=self.sess)

  def test_generate_np_can_be_called_with_different_decay_factor(self):
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    for decay_factor in [0.0, 0.5, 1.0]:
      x_adv = self.attack.generate_np(x_val, eps=0.5, ord=np.inf,
                                      decay_factor=decay_factor,
                                      clip_min=-5.0, clip_max=5.0)

      delta = np.max(np.abs(x_adv - x_val), axis=1)
      self.assertClose(delta, 0.5)

  def test_multiple_initial_random_step(self):
    # There is no initial random step, so nothing to test here
    pass


class TestMadryEtAl(CleverHansTest):
  def setUp(self):
    super(TestMadryEtAl, self).setUp()
    self.model = DummyModel('madryetal_dummy_model')
    self.sess = tf.Session()

  def test_attack_can_be_constructed(self):
    # The test passes if this does not raise an exception
    self.attack = MadryEtAl(self.model, sess=self.sess)


class TestBasicIterativeMethod(CleverHansTest):
  def setUp(self):
    super(TestBasicIterativeMethod, self).setUp()
    self.model = DummyModel('bim_dummy_model')
    self.sess = tf.Session()

  def test_attack_can_be_constructed(self):
    # The test passes if this raises no exceptions
    self.attack = BasicIterativeMethod(self.model, sess=self.sess)


class TestFastFeatureAdversaries(CleverHansTest):
  def setUp(self):
    super(TestFastFeatureAdversaries, self).setUp()

    def make_imagenet_cnn(input_shape=(None, 224, 224, 3)):
      """
      Similar CNN to AlexNet.
      """

      class ModelImageNetCNN(Model):
        def __init__(self, scope, nb_classes=1000, **kwargs):
          del kwargs
          Model.__init__(self, scope, nb_classes, locals())

        def fprop(self, x, **kwargs):
          del kwargs
          my_conv = functools.partial(tf.layers.conv2d,
                                      kernel_size=3,
                                      strides=2,
                                      padding='valid',
                                      activation=tf.nn.relu,
                                      kernel_initializer=HeReLuNormalInitializer)
          my_dense = functools.partial(tf.layers.dense,
                                       kernel_initializer=HeReLuNormalInitializer)
          with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            for depth in [96, 256, 384, 384, 256]:
              x = my_conv(x, depth)
            y = tf.layers.flatten(x)
            y = my_dense(y, 4096, tf.nn.relu)
            y = fc7 = my_dense(y, 4096, tf.nn.relu)
            y = my_dense(y, 1000)
            return {'fc7': fc7,
                    self.O_LOGITS: y,
                    self.O_PROBS: tf.nn.softmax(logits=y)}

      return ModelImageNetCNN('imagenet')

    self.input_shape = [10, 224, 224, 3]
    self.sess = tf.Session()
    self.model = make_imagenet_cnn(self.input_shape)
    self.attack = FastFeatureAdversaries(self.model, sess=self.sess)

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
    attack_params = {'eps': 5. / 256, 'clip_min': 0., 'clip_max': 1.,
                     'nb_iter': 10, 'eps_iter': 0.005,
                     'layer': layer}
    x_adv = self.attack.generate(x_src, x_guide, **attack_params)
    h_adv = self.model.fprop(x_adv)[layer]
    h_src = self.model.fprop(x_src)[layer]
    h_guide = self.model.fprop(x_guide)[layer]

    init = tf.global_variables_initializer()
    self.sess.run(init)

    ha, hs, hg, _xa, _xs, _xg = self.sess.run(
        [h_adv, h_src, h_guide, x_adv, x_src, x_guide])
    d_as = np.sqrt(((hs - ha) * (hs - ha)).sum())
    d_ag = np.sqrt(((hg - ha) * (hg - ha)).sum())
    d_sg = np.sqrt(((hg - hs) * (hg - hs)).sum())
    print("L2 distance between source and adversarial example `%s`: %.4f" %
          (layer, d_as))
    print("L2 distance between guide and adversarial example `%s`: %.4f" %
          (layer, d_ag))
    print("L2 distance between source and guide `%s`: %.4f" %
          (layer, d_sg))
    print("d_ag/d_sg*100 `%s`: %.4f" % (layer, d_ag * 100 / d_sg))
    self.assertTrue(d_ag * 100 / d_sg < 50.)


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

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

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
    self.assertEqual(x_val.shape, x_adv_p.shape)
    x_adv = self.sess.run(x_adv_p, {x: x_val, y: feed_labs})

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

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


class SimpleSpatialBrightPixelModel(Model):
  """
  If there is a bright pixel in the image returns the first class.
  Otherwise returns the second class. Spatial attack should push the
  bright pixels off of the image.
  """

  def __init__(self, scope='simple_spatial', nb_classes=2, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())

  def fprop(self, x, **kwargs):
    del kwargs

    flat_x = slim.flatten(x)
    first_logit = tf.reduce_max(flat_x, axis=1)
    second_logit = tf.ones_like(first_logit) * 0.5
    res = tf.stack([second_logit, first_logit], axis=1)
    return {self.O_LOGITS: res,
            self.O_PROBS: tf.nn.softmax(res)}


@unittest.skipIf(
    [int(v) for v in tf.__version__.split('.')[:2]] < [1, 6],
    "SpatialAttack requires tf 1.6 or higher")
class TestSpatialTransformationMethod(CleverHansTest):
  """Tests for SpatialTransformationMethod"""

  def setUp(self):
    """
    Allocate session, model, and attack + initialize tf Variables
    """
    super(TestSpatialTransformationMethod, self).setUp()

    self.sess = tf.Session()
    self.model = SimpleSpatialBrightPixelModel()
    self.attack = SpatialTransformationMethod(self.model, sess=self.sess)

    # initialize model
    with tf.name_scope('dummy_model_spatial'):
      self.model.get_logits(tf.placeholder(tf.float32, shape=(None, 2, 2, 1)))
    self.sess.run(tf.global_variables_initializer())

  def test_no_transformation(self):
    """Test that setting transformation params to 0. is a no-op"""
    x_val = np.random.rand(100, 2, 2, 1)
    x_val = np.array(x_val, dtype=np.float32)
    x = tf.placeholder(tf.float32, shape=(None, 2, 2, 1))

    x_adv_p = self.attack.generate(x, batch_size=100, dx_min=0.0,
                                   dx_max=0.0, n_dxs=1, dy_min=0.0,
                                   dy_max=0.0, n_dys=1, angle_min=0,
                                   angle_max=0, n_angles=1)
    x_adv = self.sess.run(x_adv_p, {x: x_val})
    self.assertClose(x_adv, x_val)

  def test_push_pixels_off_image(self):
    """Test that the attack pushes some pixels off the image"""
    x_val = np.random.rand(100, 2, 2, 1)
    x_val = np.array(x_val, dtype=np.float32)

    # The correct answer is that they are bright
    # So the attack must push the pixels off the edge
    y = np.zeros([100, 2])
    y[:, 0] = 1.

    x = tf.placeholder(tf.float32, shape=(None, 2, 2, 1))
    x_adv_p = self.attack.generate(x,
                                   y=y, batch_size=100, dx_min=-0.5,
                                   dx_max=0.5, n_dxs=3, dy_min=-0.5,
                                   dy_max=0.5, n_dys=3, angle_min=0,
                                   angle_max=0, n_angles=1)
    x_adv = self.sess.run(x_adv_p, {x: x_val})

    old_labs = np.argmax(y, axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    print(np.mean(old_labs == new_labs))
    self.assertTrue(np.mean(old_labs == new_labs) < 0.3)

  def test_keep_pixels_on_image(self):
    """Test that the attack does not push some pixels off the image"""
    x_val = np.random.rand(100, 2, 2, 1)
    x_val = np.array(x_val, dtype=np.float32)

    # The correct answer is that they are NOT bright
    # So the attack must NOT push the pixels off the edge
    y = np.zeros([100, 2])
    y[:, 0] = 1.

    x = tf.placeholder(tf.float32, shape=(None, 2, 2, 1))
    x_adv_p = self.attack.generate(x,
                                   y=y, batch_size=100, dx_min=-0.5,
                                   dx_max=0.5, n_dxs=3, dy_min=-0.5,
                                   dy_max=0.5, n_dys=3, angle_min=0,
                                   angle_max=0, n_angles=1)
    x_adv = self.sess.run(x_adv_p, {x: x_val})

    old_labs = np.argmax(y, axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    print(np.mean(old_labs == new_labs))
    self.assertTrue(np.mean(old_labs == new_labs) < 0.3)


class TestHopSkipJumpAttack(CleverHansTest):
  """Tests for Test HopSkipJumpAttack"""

  def setUp(self):
    super(TestHopSkipJumpAttack, self).setUp()

    self.sess = tf.Session()
    self.model = SimpleModel()
    self.attack = HopSkipJumpAttack(self.model, sess=self.sess)

  def test_generate_np_untargeted_l2(self):
    x_val = np.random.rand(50, 2)
    x_val = np.array(x_val, dtype=np.float32)
    bapp_params = {
        'constraint': 'l2',
        'stepsize_search': 'geometric_progression',
        'num_iterations': 10,
        'verbose': True,
    }
    x_adv = self.attack.generate_np(x_val, **bapp_params)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

  def test_generate_untargeted_linf(self):

    x_val = np.random.rand(50, 2)
    x_val = np.array(x_val, dtype=np.float32)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)

    # Requires input to have batchsize 1.
    x = tf.placeholder(tf.float32, [1, 2])

    bapp_params = {
        'constraint': 'linf',
        'stepsize_search': 'grid_search',
        'num_iterations': 10,
        'verbose': True,
    }
    x_adv_p = self.attack.generate(x, **bapp_params)

    self.assertEqual(x_adv_p.shape, [1, 2])
    x_adv = []
    for single_x_val in x_val:
      single_x_adv = self.sess.run(
          x_adv_p, {x: np.expand_dims(single_x_val, 0)})
      x_adv.append(single_x_adv)

    x_adv = np.concatenate(x_adv, axis=0)

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

  def test_generate_np_targeted_linf(self):
    x_val = np.random.rand(200, 2)
    x_val = np.array(x_val, dtype=np.float32)

    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    x_val_pos = x_val[orig_labs == 1]
    x_val_neg = x_val[orig_labs == 0]

    x_val_under_attack = np.concatenate(
        (x_val_pos[:25], x_val_neg[:25]), axis=0)
    y_target = np.eye(2)[np.concatenate(
        (np.zeros(25), np.ones(25))).astype(int)]
    image_target = np.concatenate((x_val_neg[25:50], x_val_pos[25:50]), axis=0)

    bapp_params = {
        'constraint': 'linf',
        'stepsize_search': 'geometric_progression',
        'num_iterations': 10,
        'verbose': True,
        'y_target': y_target,
        'image_target': image_target,
    }
    x_adv = self.attack.generate_np(x_val_under_attack, **bapp_params)

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)

    self.assertTrue(np.mean(np.argmax(y_target, axis=1) == new_labs)
                    > 0.9)

  def test_generate_targeted_l2(self):

    # Create data in numpy arrays.
    x_val = np.random.rand(200, 2)
    x_val = np.array(x_val, dtype=np.float32)
    orig_labs = np.argmax(self.sess.run(self.model.get_logits(x_val)), axis=1)
    x_val_pos = x_val[orig_labs == 1]
    x_val_neg = x_val[orig_labs == 0]
    x_val_under_attack = np.concatenate(
        (x_val_pos[:25], x_val_neg[:25]), axis=0)
    y_target = np.eye(2)[np.concatenate(
        (np.zeros(25), np.ones(25))).astype(int)]
    image_target = np.concatenate((x_val_neg[25:50], x_val_pos[25:50]), axis=0)

    # Create placeholders.
    # Require input has batchsize 1.
    x = tf.placeholder(tf.float32, [1, 2])
    y_target_ph = tf.placeholder(tf.float32, [1, 2])
    image_target_ph = tf.placeholder(tf.float32, [1, 2])

    # Create graph.
    bapp_params = {
        'constraint': 'l2',
        'stepsize_search': 'grid_search',
        'num_iterations': 10,
        'verbose': True,
        'y_target': y_target_ph,
        'image_target': image_target_ph,
    }
    x_adv_p = self.attack.generate(x, **bapp_params)
    self.assertEqual(x_adv_p.shape, [1, 2])

    # Generate adversarial examples.
    x_adv = []
    for i, single_x_val in enumerate(x_val_under_attack):
      print(image_target.shape, y_target.shape)
      single_x_adv = self.sess.run(x_adv_p,
                                   {x: np.expand_dims(single_x_val, 0),
                                    y_target_ph: np.expand_dims(y_target[i], 0),
                                    image_target_ph: np.expand_dims(image_target[i], 0)})
      x_adv.append(single_x_adv)
    x_adv = np.concatenate(x_adv, axis=0)

    new_labs = np.argmax(self.sess.run(self.model.get_logits(x_adv)), axis=1)
    self.assertTrue(np.mean(np.argmax(y_target, axis=1) == new_labs)
                    > 0.9)