# pylint: disable=missing-docstring
import numpy as np
import torch

import cleverhans.future.torch.utils as utils
from cleverhans.devtools.checks import CleverHansTest

class TestOptimizeLinear(CleverHansTest):
  """
  Identical to the TestOptimizeLinear in tests_tf/test_attacks.
  """
  def test_optimize_linear_linf(self):
    grad = torch.tensor([[1., -2.]])
    eta = utils.optimize_linear(grad, eps=1., norm=np.inf)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, grad.abs().sum())
    self.assertClose(eta.abs(), 1.)

  def test_optimize_linear_l2(self):
    grad = torch.tensor([[.5 ** .5, -.5 ** .5]])
    eta = utils.optimize_linear(grad, eps=1., norm=2)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 1.)
    self.assertClose(eta.pow(2).sum().sqrt(), 1.)

  def test_optimize_linear_l1(self):
    grad = torch.tensor([[1., -2.]])
    eta = utils.optimize_linear(grad, eps=1., norm=1)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 2.)
    self.assertClose(eta.abs().sum(), 1.)

  def test_optimize_linear_l1_ties(self):
    grad = torch.tensor([[2., -2.]])
    eta = utils.optimize_linear(grad, eps=1., norm=1)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 2.)
    self.assertClose(eta.abs().sum(), 1.)
