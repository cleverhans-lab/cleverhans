# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

  def test_attack_can_be_called_with_different_settings(self):
    raise NotImplementedError('must be implemented by a subclass')

class TestFastGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestFastGradientMethod, self).setUp()
    self.attack = fast_gradient_method

class TestProjectedGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestProjectedGradientMethod, self).setUp()
    self.attack = projected_gradient_descent
