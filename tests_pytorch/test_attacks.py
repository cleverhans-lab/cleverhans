# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
from nose.plugins.skip import SkipTest
import torch
import torch.nn.functional as F

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.future.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent

class SimpleModel(torch.nn.Module):

  def __init__(self, n_in, n_hidden, n_out):
    super(SimpleModel, self).__init__()
    self.fc1 = torch.nn.Linear(n_in, n_hidden)
    self.fc2 = torch.nn.Linear(n_hidden, n_out)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x

class CommonAttackProperties(CleverHansTest):

  def setUp(self):
    # pylint: disable=unidiomatic-typecheck
    if type(self) is CommonAttackProperties:
      raise SkipTest()

    super(CommonAttackProperties, self).setUp()
    dim1 = 2
    self.model = SimpleModel(n_in=dim1, n_hidden=5, n_out=2)
    self.x = torch.randn(100, dim1)
    self.normalized_x = torch.rand(100, dim1) # truncated between [0, 1)
    self.red_ind = list(range(1, len(self.x.size())))

  def help_adv_examples_success_rate(self, **kwargs):
    x_adv = self.attack(**kwargs)
    _, ori_label = self.model(self.x).max(1)
    _, adv_label = self.model(x_adv).max(1)
    adv_acc = adv_label.eq(ori_label).sum().to(torch.float) / self.x.size(0)
    self.assertLess(adv_acc, .5)

  def help_targeted_adv_examples_success_rate(self, **kwargs):
    y_target = torch.randint(low=0, high=2, size=(self.x.size(0),)) # a two-class problem
    try:
      x_adv = self.attack(y=y_target, targeted=True, **kwargs)
    except NotImplementedError:
      raise SkipTest()

    _, adv_label = self.model(x_adv).max(1)
    adv_success = adv_label.eq(y_target).sum().to(torch.float) / self.x.size(0)
    self.assertGreater(adv_success, .7)

class TestFastGradientMethod(CommonAttackProperties):
  """
  Tests for FGM. Note that the core method of FGM, the optimize_linear function, is actually
  tested in test_utils. So there's no need for repeating those tests here.
  """

  def setUp(self):
    super(TestFastGradientMethod, self).setUp()
    self.attack = fast_gradient_method
    self.ord_list = [1, 2, np.inf]
    self.eps_list = [0, .1, .3, 1., 3]

  def test_invalid_input(self):
    x = torch.tensor([[-2., 3.]])
    for ord in self.ord_list:
      self.assertRaises(
          AssertionError, self.attack, model_fn=self.model, x=x, eps=.1,
          ord=ord, clip_min=-1., clip_max=1., sanity_checks=True
          )

  def test_invalid_eps(self):
    for ord in self.ord_list:
      self.assertRaises(AssertionError, self.attack, model_fn=self.model, x=self.x, eps=-.1, ord=ord)

  def test_eps_equals_zero(self):
    for ord in self.ord_list:
      self.assertClose(self.attack(model_fn=self.model, x=self.x, eps=0, ord=ord), self.x)

  def test_eps(self):
    # test if the attack respects the norm constraint
    # NOTE this has been tested with the optimize_linear function in test_utils, so duplicate tests are not needed here.
    # Although, if ever switch the engine of the FGM method to some
    # function other than optimize_linear. This test should be added.
    raise SkipTest()

  def test_clips(self):
    clip_min = -1.
    clip_max = 1.
    for ord in self.ord_list:
      x_adv = self.attack(
          model_fn=self.model, x=self.normalized_x, eps=.3, ord=ord,
          clip_min=clip_min, clip_max=clip_max
          )
      self.assertTrue(torch.all(x_adv <= clip_max))
      self.assertTrue(torch.all(x_adv >= clip_min))

  def test_invalid_clips(self):
    clip_min = .5
    clip_max = -.5
    for ord in self.ord_list:
      self.assertRaises(
          AssertionError, self.attack, model_fn=self.model, x=self.x, eps=.1,
          ord=ord, clip_min=clip_min, clip_max=clip_max
          )

  def test_adv_example_success_rate_linf(self):
    # use normalized_x to make sure the same eps gives uniformly high attack
    # success rate across randomized tests
    self.help_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=np.inf, eps=.5)

  def test_targeted_adv_example_success_rate_linf(self):
    self.help_targeted_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=np.inf, eps=.5)

  def test_adv_example_success_rate_l1(self):
    self.help_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=1, eps=.5)

  def test_targeted_adv_example_success_rate_l1(self):
    self.help_targeted_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=1, eps=.5)

  def test_adv_example_success_rate_l2(self):
    self.help_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=2, eps=.5)

  def test_targeted_adv_example_success_rate_l2(self):
    self.help_targeted_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=2, eps=.5)

class TestProjectedGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestProjectedGradientMethod, self).setUp()
    self.attack = projected_gradient_descent

  def test_adv_example_success_rate_linf(self):
    # use normalized_x to make sure the same eps gives uniformly high attack
    # success rate across randomized tests
    self.help_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=np.inf,
        eps=.5, eps_iter=.05, nb_iter=10)

  def test_targeted_adv_example_success_rate_linf(self):
    self.help_targeted_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=np.inf,
        eps=.5, eps_iter=.05, nb_iter=10)

  def test_adv_example_success_rate_l1(self):
    self.help_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=1,
        eps=.5, eps_iter=.05, nb_iter=10)

  def test_targeted_adv_example_success_rate_l1(self):
    self.help_targeted_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=1,
        eps=.5, eps_iter=.05, nb_iter=10)

  def test_adv_example_success_rate_l2(self):
    self.help_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=2,
        eps=.5, eps_iter=.05, nb_iter=10)

  def test_targeted_adv_example_success_rate_l2(self):
    self.help_targeted_adv_examples_success_rate(
        model_fn=self.model, x=self.normalized_x, ord=2,
        eps=.5, eps_iter=.05, nb_iter=10)

if __name__ == "__main__":
  unittest.main()
