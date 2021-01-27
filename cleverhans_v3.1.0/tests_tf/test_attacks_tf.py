"""Tests of cleverhans.attacks_tf

"""
# pylint: disable=missing-docstring
from functools import partial
import unittest

import numpy as np
import tensorflow as tf

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.attacks_tf import fgm, pgd_attack, \
    UnrolledAdam, UnrolledGradientDescent, parallel_apply_transformations
from cleverhans.devtools.mocks import random_feed_dict
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
      w1 = tf.constant(
          [[1.5, .3], [-2, 0.3]], dtype=tf.as_dtype(x.dtype))
      w2 = tf.constant(
          [[-2.4, 1.2], [0.5, -2.3]], dtype=tf.as_dtype(x.dtype))
    h1 = tf.nn.sigmoid(tf.matmul(x, w1))
    res = tf.matmul(h1, w2)
    return {self.O_LOGITS: res, self.O_PROBS: tf.nn.softmax(res)}


class TestAttackTF(CleverHansTest):
  def setUp(self):
    super(TestAttackTF, self).setUp()
    self.sess = tf.Session()
    self.model = SimpleModel()

  def test_fgm_gradient_max(self):
    input_dim = 2
    nb_classes = 3
    batch_size = 4
    rng = np.random.RandomState([2017, 8, 23])
    x = tf.placeholder(tf.float32, [batch_size, input_dim])
    weights = tf.placeholder(tf.float32, [input_dim, nb_classes])
    logits = tf.matmul(x, weights)
    probs = tf.nn.softmax(logits)
    adv_x = fgm(x, probs)
    random_example = rng.randint(batch_size)
    random_feature = rng.randint(input_dim)
    output = tf.slice(adv_x, [random_example, random_feature], [1, 1])
    dx, = tf.gradients(output, x)
    # The following line catches GitHub issue #243
    self.assertIsNotNone(dx)
    dx = self.sess.run(dx, feed_dict=random_feed_dict(rng, [x, weights]))
    ground_truth = np.zeros((batch_size, input_dim))
    ground_truth[random_example, random_feature] = 1.
    self.assertClose(dx, ground_truth)

  def helper_pgd_attack(self,
                        unrolled_optimizer,
                        targeted,
                        nb_iters=20,
                        epsilon=.5,
                        clip_min=-5.,
                        clip_max=5.,
                        assert_threshold=0.5):
    def loss_fn(input_image, label, targeted):
      res = self.model.fprop(input_image)
      logits = res[self.model.O_LOGITS]
      multiplier = 1. if targeted else -1.
      return multiplier * tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=label, logits=logits)

    x_val_ph = tf.placeholder(tf.float32, shape=[100, 2])
    x_val = np.random.randn(100, 2).astype(np.float32)
    init_model_output = self.model.fprop(x_val_ph)
    init_model_logits = init_model_output[self.model.O_LOGITS]
    if targeted:
      labels = np.random.random_integers(0, 1, size=(100, ))
    else:

      labels = tf.stop_gradient(tf.argmax(init_model_logits, axis=1))

    def _project_perturbation(perturbation, epsilon, input_image,
                              clip_min, clip_max):
      clipped_perturbation = tf.clip_by_value(perturbation, -epsilon,
                                              epsilon)
      new_image = tf.clip_by_value(input_image + clipped_perturbation,
                                   clip_min, clip_max)
      return new_image - input_image

    x_adv = pgd_attack(
        loss_fn=partial(loss_fn, targeted=targeted),
        input_image=x_val_ph,
        label=labels,
        epsilon=epsilon,
        num_steps=nb_iters,
        optimizer=unrolled_optimizer,
        project_perturbation=_project_perturbation,
        clip_min=clip_min, clip_max=clip_max)

    final_model_output = self.model.fprop(x_adv)
    final_model_logits = final_model_output[self.model.O_LOGITS]

    if not targeted:
      logits1, logits2 = self.sess.run(
          [init_model_logits, final_model_logits],
          feed_dict={x_val_ph: x_val})
      preds1 = np.argmax(logits1, axis=1)
      preds2 = np.argmax(logits2, axis=1)

      self.assertTrue(
          np.mean(preds1 == preds2) < assert_threshold,
          np.mean(preds1 == preds2))

    else:
      logits_adv = self.sess.run(
          final_model_logits, feed_dict={x_val_ph: x_val})
      preds_adv = np.argmax(logits_adv, axis=1)

      self.assertTrue(np.mean(labels == preds_adv) > assert_threshold)

  def test_pgd_untargeted_attack_with_adam_optimizer(self):
    unrolled_optimizer = UnrolledAdam(lr=0.1)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=False,
        epsilon=.5,
        nb_iters=20,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.7)

  def test_stronger_pgd_untargeted_attack_with_adam_optimizer(self):
    unrolled_optimizer = UnrolledAdam(lr=0.1)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=False,
        epsilon=5.,
        nb_iters=100,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.1)

  def test_pgd_targeted_attack_with_adam_optimizer(self):
    unrolled_optimizer = UnrolledAdam(lr=0.1)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=True,
        epsilon=.5,
        nb_iters=20,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.7)

  def test_stronger_pgd_targeted_attack_with_adam_optimizer(self):
    unrolled_optimizer = UnrolledAdam(lr=0.1)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=True,
        epsilon=5.,
        nb_iters=100,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.9)

  def test_pgd_untargeted_attack_with_sgd_optimizer(self):
    unrolled_optimizer = UnrolledGradientDescent(lr=1000.)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=False,
        epsilon=.5,
        nb_iters=20,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.6)

  def test_stronger_pgd_untargeted_attack_with_sgd_optimizer(self):
    unrolled_optimizer = UnrolledGradientDescent(lr=1000.)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=False,
        epsilon=5.,
        nb_iters=100,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.1)

  def test_pgd_targeted_attack_with_sgd_optimizer(self):
    unrolled_optimizer = UnrolledGradientDescent(lr=1000.)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=True,
        epsilon=.5,
        nb_iters=20,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.6)

  def test_stronger_pgd_targeted_attack_with_sgd_optimizer(self):
    unrolled_optimizer = UnrolledGradientDescent(lr=1000.)
    self.helper_pgd_attack(
        unrolled_optimizer=unrolled_optimizer,
        targeted=True,
        epsilon=5.,
        nb_iters=100,
        clip_min=-10.,
        clip_max=10.,
        assert_threshold=0.9)

  @unittest.skip("This test requires human inspection of the images")
  def test_parallel_apply(self):
    def _save_image_to_png(image_np, filename):
      from PIL import Image
      import os

      dirname = os.path.dirname(filename)
      if not os.path.exists(dirname):
        os.makedirs(dirname)

      if image_np.shape[-1] == 3:
        img = Image.fromarray(np.uint8(image_np * 255.), 'RGB')
      else:
        img = Image.fromarray(np.uint8(image_np[:, :, 0] * 255.), 'L')
      img.save(filename)

    x = tf.ones([3, 200, 200, 3])

    transforms = [
        [0.2, 0, 20],
        [0, 0, 0],
        # [-0.2, 0, 20],
        # [-0.4, 0, 20],
    ]
    transformed_ims = parallel_apply_transformations(
        x, transforms, black_border_size=30)

    worst_sample_idx = tf.convert_to_tensor([0, 1, 1])
    batch_size = tf.shape(x)[0]
    keys = tf.stack([
        tf.range(batch_size, dtype=tf.int32),
        tf.cast(worst_sample_idx, tf.int32)
    ], axis=1)

    transformed_ims_bshwc = tf.einsum('sbhwc->bshwc', transformed_ims)
    after_lookup = tf.gather_nd(transformed_ims_bshwc, keys)  # BHWC

    with tf.Session() as sess:
      img_batch_np = sess.run(after_lookup)[:, :, :, :]

    for i, img in enumerate(img_batch_np):
      filename = "/tmp/test_image%s.png" % (i)
      _save_image_to_png(img, filename)


if __name__ == '__main__':
  unittest.main()
