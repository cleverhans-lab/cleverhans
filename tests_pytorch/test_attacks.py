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
    self.model = SimpleModel(n_in=2, n_hidden=3, n_out=2)
    self.x = torch.randn(10, 2)

class TestFastGradientMethod(CommonAttackProperties):
  """
  Tests for FGM. Note that the core method of FGM, the optimize_linear function, is actually
  tested in test_utils. So there's no need for repeating those tests here.
  """

  def setUp(self):
    super(TestFastGradientMethod, self).setUp()
    self.attack = fast_gradient_method
    self.ord_list = [1, 2, np.inf]

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
    pass

  def test_clips(self):
    normalized_x = torch.rand(10, 2)
    clip_min = -1.
    clip_max = 1.
    for ord in self.ord_list:
      x_adv = self.attack(
          model_fn=self.model, x=normalized_x, eps=.3, ord=ord,
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

class TestProjectedGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestProjectedGradientMethod, self).setUp()
    self.attack = projected_gradient_descent

if __name__ == "__main__":
  unittest.main()
