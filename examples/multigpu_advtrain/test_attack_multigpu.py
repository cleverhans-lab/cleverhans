from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from attacks_multigpu import MadryEtAlMultiGPU
from model import MLPnGPU
from model import LayernGPU


import sys
import os
sys.path.insert(0, os.path.abspath('../../tests_tf/'))
from test_attacks import TestMadryEtAl  # NOQA


class TestMadryEtAlMultiGPU(TestMadryEtAl):
  """
  By inherting from `TestMadryEtAl`, the attack `MadryEtAlMultiGPU` can be
  tested against all tests of the base attack.
  """

  def setUp(self):
    super(TestMadryEtAlMultiGPU, self).setUp()

    class SimpleLayer(LayernGPU):

      def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.W1 = tf.constant([[1.5, .3], [-2, 0.3]], dtype=tf.float32)
        self.W2 = tf.constant([[-2.4, 1.2], [0.5, -2.3]],
                              dtype=tf.float32)

      def fprop_noscope(self, x):
        h1 = tf.nn.sigmoid(tf.matmul(x, self.W1))
        res = tf.matmul(h1, self.W2)
        return res

    input_shape = (None, 2)
    self.model_ngpu = MLPnGPU([SimpleLayer()], input_shape)

    self.attack_single_gpu = self.attack
    self.attack_multi_gpu = MadryEtAlMultiGPU(self.model_ngpu,
                                              sess=self.sess)
    self.attack = self.attack_multi_gpu

  def test_single_vs_multi_gpu(self):
    """
    Compare the strength of the single GPU and multi-GPU implementations.
    """
    x_val = np.random.rand(100, 2)
    x_val = np.array(x_val, dtype=np.float32)

    def multi_attack(attack):
      flags = {'ngpu': 1, 'eps': 1.0, 'eps_iter': 0.01,
               'clip_min': 0.5, 'clip_max': 0.7, 'nb_iter': 2,
               'rand_init': True}

      orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
      new_labs_multi = orig_labs.copy()
      # Generate multiple adversarial examples
      for i in range(40):
        x_adv = attack.generate_np(x_val, **flags)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

        # Examples for which we have not found adversarial examples
        indices = (orig_labs == new_labs_multi)
        new_labs_multi[indices] = new_labs[indices]

      return np.mean(orig_labs == new_labs_multi)

    acc_s = multi_attack(self.attack_single_gpu)
    acc_m = multi_attack(self.attack_multi_gpu)

    self.assertClose(acc_s, acc_m, atol=1e-2)


if __name__ == '__main__':
  unittest.main()
