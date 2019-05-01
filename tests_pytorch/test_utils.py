# pylint: disable=missing-docstring
import numpy as np
import torch

from cleverhans import utils_pytorch
from cleverhans.devtools.checks import CleverHansTest

class TestOptimizeLinear(CleverHansTest):
  """
  Identical to the TestOptimizeLinear in tests_tf/test_attacks.
  """
  def test_optimize_linear_linf(self):
    grad = torch.tensor([[1., -2.]])
    eta = utils_pytorch.optimize_linear(grad, eps=1., ord=np.inf)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, grad.abs().sum())
    self.assertClose(eta.abs(), 1.)

  def test_optimize_linear_l2(self):
    grad = torch.tensor([[.5 ** .5, -.5 ** .5]])
    eta = utils_pytorch.optimize_linear(grad, eps=1., ord=2)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 1.)
    self.assertClose(eta.pow(2).sum().sqrt(), 1.)

  def test_optimize_linear_l1(self):
    grad = torch.tensor([[1., -2.]])
    eta = utils_pytorch.optimize_linear(grad, eps=1., ord=1)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 2.)
    self.assertClose(eta.abs().sum(), 1.)

  def test_optimize_linear_l1_ties(self):
    grad = torch.tensor([[2., -2.]])
    eta = utils_pytorch.optimize_linear(grad, eps=1., ord=1)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 2.)
    self.assertClose(eta.abs().sum(), 1.)
