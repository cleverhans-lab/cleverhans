# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf

from cleverhans.devtools.checks import CleverHansTest

from runner import RunnerMultiGPU


class TestRunnerMultiGPU(CleverHansTest):
  def setUp(self):
    super(TestRunnerMultiGPU, self).setUp()
    self.sess = tf.Session()

    inputs = []
    outputs = []
    self.niter = 10
    niter = self.niter
    # A Simple graph with `niter` sub-graphs.
    with tf.variable_scope(None, 'runner'):
      for i in range(niter):
        v = tf.get_variable('v%d' % i, shape=(100, 10))
        w = tf.get_variable('w%d' % i, shape=(100, 1))

        inputs += [{'v': v, 'w': w}]
        outputs += [{'v': v, 'w': w}]

    self.runner = RunnerMultiGPU(inputs, outputs, sess=self.sess)

  def help_test_runner(self, ninputs, niter):
    """
    Tests the MultiGPU runner by feeding in random Tensors for `ninputs`
    steps. Then validating the output after `niter-1` steps.
    """
    v_val = []
    w_val = []
    for i in range(ninputs):
      v_val += [np.random.rand(100, 10)]
      w_val += [np.random.rand(100, 1)]
      fvals = self.runner.run({'v': v_val[i], 'w': w_val[i]})
      self.assertTrue(len(fvals) == 0)
      self.assertFalse(self.runner.is_finished())

    for i in range(niter-ninputs-1):
      self.assertFalse(self.runner.is_finished())
      fvals = self.runner.run()
      self.assertTrue(len(fvals) == 0)
      self.assertFalse(self.runner.is_finished())

    for i in range(ninputs):
      self.assertFalse(self.runner.is_finished())
      fvals = self.runner.run()
      self.assertTrue('v' in fvals and 'w' in fvals)
      self.assertTrue(np.allclose(fvals['v'], v_val[i]))
      self.assertTrue(np.allclose(fvals['w'], w_val[i]))

    self.assertTrue(self.runner.is_finished())

  def test_queue_full(self):
    self.help_test_runner(self.niter-1, self.niter)

  def test_queue_half(self):
    self.help_test_runner(self.niter//2, self.niter)


if __name__ == '__main__':
  unittest.main()
