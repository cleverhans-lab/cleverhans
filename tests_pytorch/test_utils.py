# pylint: disable=missing-docstring
import numpy as np
import torch

from cleverhans import utils_pytorch
from cleverhans.devtools.checks import CleverHansTest

class TestOptimizeLinear(CleverHansTest):
  """
  Identical to the TestOptimizeLinear in tests_tf/test_attacks.
  """
  def setUp(self):
    super(TestOptimizeLinear, self).setUp()
    self.fn = utils_pytorch.optimize_linear
    self.rand_grad = torch.rand(10, 3, 2)
    self.red_ind = list(range(1, len(self.rand_grad.size())))
    # eps needs to be nonnegative
    self.eps_list = [0, .1, 1., 3]

  def test_optimize_linear_linf(self):
    grad = torch.tensor([[1., -2.]])
    eta = self.fn(grad, eps=1., ord=np.inf)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, grad.abs().sum())
    self.assertClose(eta.abs(), 1.)

  def test_optimize_linear_l2(self):
    grad = torch.tensor([[.5 ** .5, -.5 ** .5]])
    eta = self.fn(grad, eps=1., ord=2)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 1.)
    self.assertClose(eta.pow(2).sum().sqrt(), 1.)

  def test_optimize_linear_l1(self):
    grad = torch.tensor([[1., -2.]])
    eta = self.fn(grad, eps=1., ord=1)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 2.)
    self.assertClose(eta.abs().sum(), 1.)

  def test_optimize_linear_l1_ties(self):
    grad = torch.tensor([[2., -2.]])
    eta = self.fn(grad, eps=1., ord=1)
    objective = torch.sum(grad * eta)

    self.assertEqual(grad.size(), eta.size())
    self.assertClose(objective, 2.)
    self.assertClose(eta.abs().sum(), 1.)

  def test_optimize_linear_linf_satisfies_norm_constraint(self):
    for eps in self.eps_list:
      eta = self.fn(self.rand_grad, eps=eps, ord=np.inf)
      self.assertClose(eta.abs(), eps)

  def test_optimize_linear_l1_satisfies_norm_constraint(self):
    for eps in self.eps_list:
      eta = self.fn(self.rand_grad, eps=eps, ord=1)
      norm = eta.abs().sum(dim=self.red_ind)
      self.assertTrue(torch.all(norm == eps * torch.ones_like(norm)))

  def test_optimize_linear_l2_satisfies_norm_constraint(self):
    for eps in self.eps_list:
      eta = self.fn(self.rand_grad, eps=eps, ord=2)
      avoid_zero_div = torch.tensor(1e-12)
      square = torch.max(
          avoid_zero_div,
          torch.sum(self.rand_grad ** 2, self.red_ind, keepdim=True)
          )
      norm = eta.pow(2).sum(dim=self.red_ind, keepdim=True).sqrt()
      one_mask = (square <= avoid_zero_div).to(torch.float) * norm + \
              (square > avoid_zero_div).to(torch.float)
      self.assertTrue(torch.allclose(norm, eps * one_mask, rtol=1e-05, atol=1e-08))
