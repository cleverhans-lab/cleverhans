# pylint: disable=missing-docstring
import numpy as np
import torch

import cleverhans.future.torch.utils as utils
from cleverhans.devtools.checks import CleverHansTest

class TestOptimizeLinear(CleverHansTest):
  """
  Identical to the TestOptimizeLinear in tests_tf/test_attacks.
  """
  def setUp(self):
    super(TestOptimizeLinear, self).setUp()
    self.clip_eta = utils.clip_eta
    self.rand_grad = torch.randn(100, 3, 2)
    self.rand_eta = torch.randn(100, 3, 2)
    self.red_ind = list(range(1, len(self.rand_grad.size())))
    # eps needs to be nonnegative
    self.eps_list = [0, .1, 1., 3]

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

  def test_optimize_linear_linf_satisfies_norm_constraint(self):
    for eps in self.eps_list:
      eta = utils.optimize_linear(self.rand_grad, eps=eps, norm=np.inf)
      self.assertClose(eta.abs(), eps)

  def test_optimize_linear_l1_satisfies_norm_constraint(self):
    for eps in self.eps_list:
      eta = utils.optimize_linear(self.rand_grad, eps=eps, norm=1)
      norm = eta.abs().sum(dim=self.red_ind)
      self.assertTrue(torch.allclose(norm, eps * torch.ones_like(norm)))

  def test_optimize_linear_l2_satisfies_norm_constraint(self):
    for eps in self.eps_list:
      eta = utils.optimize_linear(self.rand_grad, eps=eps, norm=2)
      # optimize_linear uses avoid_zero_div as the divisor for
      # gradients with overly small l2 norms when performing norm
      # normalizations on the gradients so as to safeguard against
      # zero division error. Therefore, the replaced gradient vectors
      # will not be l2-unit vectors after normalization. In this test,
      # these gradients are filtered out by the one_mask
      # below and are not tested.
      # NOTE the value of avoid_zero_div should be the same as the
      # avoid_zero_div used in the optimize_linear function
      avoid_zero_div = torch.tensor(1e-12)
      square = torch.max(
          avoid_zero_div,
          torch.sum(self.rand_grad ** 2, self.red_ind, keepdim=True)
          )
      norm = eta.pow(2).sum(dim=self.red_ind, keepdim=True).sqrt()
      one_mask = (
          (square <= avoid_zero_div).to(torch.float) * norm +
          (square > avoid_zero_div).to(torch.float))
      self.assertTrue(torch.allclose(norm, eps * one_mask))

  def test_clip_eta_linf(self):
    clipped = self.clip_eta(eta=self.rand_eta, norm=np.inf, eps=.5)
    self.assertTrue(torch.all(clipped <= .5))
    self.assertTrue(torch.all(clipped >= -.5))

  def test_clip_eta_l1(self):
    self.assertRaises(
        NotImplementedError, self.clip_eta, eta=self.rand_eta, norm=1, eps=.5)
    
    # TODO uncomment the actual test below after we have implemented the L1 attack
    # clipped = self.clip_eta(eta=self.rand_eta, norm=1, eps=.5)
    # norm = clipped.abs().sum(dim=self.red_ind)
    # self.assertTrue(torch.all(norm <= .5001))

  def test_clip_eta_l2(self):
    clipped = self.clip_eta(eta=self.rand_eta, norm=2, eps=.5)
    norm = clipped.pow(2).sum(dim=self.red_ind).pow(.5)
    self.assertTrue(torch.all(norm <= .5001))
