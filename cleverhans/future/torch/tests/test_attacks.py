# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from nose.plugins.skip import SkipTest
import torch

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.future.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.future.torch.attacks.spsa import spsa
from cleverhans.future.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.future.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2


class SimpleModel(torch.nn.Module):

  def __init__(self):
    super(SimpleModel, self).__init__()
    self.w1 = torch.tensor([[1.5, .3], [-2, .3]])
    self.w2 = torch.tensor([[-2.4, 1.2], [.5, -2.3]])

  def forward(self, x):
    x = torch.matmul(x, self.w1)
    x = torch.sigmoid(x)
    x = torch.matmul(x, self.w2)
    return x


class TrivialModel(torch.nn.Module):
  def __init__(self):
    super(TrivialModel, self).__init__()
    self.w1 = torch.tensor([[1.0, -1.0]])

  def forward(self, x):
    return torch.matmul(x, self.w1)


class CommonAttackProperties(CleverHansTest):

  def setUp(self):
    super(CommonAttackProperties, self).setUp()
    self.model = SimpleModel()
    self.x = torch.randn(100, 2)
    self.normalized_x = torch.rand(100, 2)  # truncated between [0, 1)
    self.red_ind = list(range(1, len(self.x.size())))
    self.ord_list = [1, 2, np.inf]

  def help_adv_examples_success_rate(self, **kwargs):
    x_adv = self.attack(model_fn=self.model, x=self.normalized_x, **kwargs)
    _, ori_label = self.model(self.normalized_x).max(1)
    _, adv_label = self.model(x_adv).max(1)
    adv_acc = (
        adv_label.eq(ori_label).sum().to(torch.float)
        / self.normalized_x.size(0))
    self.assertLess(adv_acc, .5)

  def help_targeted_adv_examples_success_rate(self, **kwargs):
    y_target = torch.randint(low=0, high=2, size=(self.normalized_x.size(0),))
    x_adv = self.attack(
        model_fn=self.model, x=self.normalized_x,
        y=y_target, targeted=True, **kwargs)

    _, adv_label = self.model(x_adv).max(1)
    adv_success = (
        adv_label.eq(y_target).sum().to(torch.float)
        / self.normalized_x.size(0))
    self.assertGreater(adv_success, .7)


class TestFastGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestFastGradientMethod, self).setUp()
    self.attack = fast_gradient_method
    self.eps_list = [0, .1, .3, 1., 3]
    self.attack_param = {
        'eps': .5,
        'clip_min': -5,
        'clip_max': 5
    }

  def test_invalid_input(self):
    x = torch.tensor([[-2., 3.]])
    for norm in self.ord_list:
      self.assertRaises(
          AssertionError, self.attack, model_fn=self.model, x=x, eps=.1,
          norm=norm, clip_min=-1., clip_max=1., sanity_checks=True
      )

  def test_invalid_eps(self):
    for norm in self.ord_list:
      self.assertRaises(
          ValueError, self.attack, model_fn=self.model,
          x=self.x, eps=-.1, norm=norm)

  def test_eps_equals_zero(self):
    for norm in self.ord_list:
      self.assertClose(
          self.attack(model_fn=self.model, x=self.x, eps=0, norm=norm),
          self.x)

  def test_eps(self):
    # test if the attack respects the norm constraint
    # NOTE this has been tested with the optimize_linear function in
    # test_utils, so duplicate tests are not needed here.
    # Although, if ever switch the engine of the FGM method to some
    # function other than optimize_linear. This test should be added.
    raise SkipTest()

  def test_clips(self):
    clip_min = -1.
    clip_max = 1.
    for norm in self.ord_list:
      x_adv = self.attack(
          model_fn=self.model, x=self.normalized_x, eps=.3, norm=norm,
          clip_min=clip_min, clip_max=clip_max
      )
      self.assertTrue(torch.all(x_adv <= clip_max))
      self.assertTrue(torch.all(x_adv >= clip_min))

  def test_invalid_clips(self):
    clip_min = .5
    clip_max = -.5
    for norm in self.ord_list:
      self.assertRaises(
          ValueError, self.attack, model_fn=self.model, x=self.x, eps=.1,
          norm=norm, clip_min=clip_min, clip_max=clip_max
      )

  def test_adv_example_success_rate_linf(self):
    # use normalized_x to make sure the same eps gives uniformly high attack
    # success rate across randomized tests
    self.help_adv_examples_success_rate(
        norm=np.inf, **self.attack_param)

  def test_targeted_adv_example_success_rate_linf(self):
    self.help_targeted_adv_examples_success_rate(
        norm=np.inf, **self.attack_param)

  def test_adv_example_success_rate_l1(self):
    self.help_adv_examples_success_rate(
        norm=1, **self.attack_param)

  def test_targeted_adv_example_success_rate_l1(self):
    self.help_targeted_adv_examples_success_rate(
        norm=1, **self.attack_param)

  def test_adv_example_success_rate_l2(self):
    self.help_adv_examples_success_rate(
        norm=2, **self.attack_param)

  def test_targeted_adv_example_success_rate_l2(self):
    self.help_targeted_adv_examples_success_rate(
        norm=2, **self.attack_param)


class TestProjectedGradientMethod(CommonAttackProperties):

  def setUp(self):
    super(TestProjectedGradientMethod, self).setUp()
    self.attack = projected_gradient_descent
    self.attack_param = {
        'eps': .5,
        'clip_min': -5,
        'clip_max': 5,
        'eps_iter': .05,
        'nb_iter': 20,
    }

  def test_invalid_input(self):
    x = torch.tensor([[-2., 3.]])
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, self.attack, model_fn=self.model, x=x, eps=.1,
            nb_iter=1, eps_iter=.05, norm=norm, clip_min=-1., clip_max=1.,
            sanity_checks=True)
      else:
        self.assertRaises(
            AssertionError, self.attack, model_fn=self.model, x=x, eps=.1,
            nb_iter=1, eps_iter=.05, norm=norm, clip_min=-1., clip_max=1.,
            sanity_checks=True)

  def test_invalid_eps(self):
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, self.attack, model_fn=self.model,
            x=self.x, eps=-.1, norm=norm, nb_iter=1, eps_iter=.01)
      else:
        self.assertRaises(
            ValueError, self.attack, model_fn=self.model,
            x=self.x, eps=-.1, norm=norm, nb_iter=1, eps_iter=.01)

  def test_invalid_eps_iter(self):
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, self.attack, model_fn=self.model,
            x=self.x, eps=.1, norm=norm, nb_iter=1, eps_iter=-.01)
      else:
        self.assertRaises(
            ValueError, self.attack, model_fn=self.model,
            x=self.x, eps=.1, norm=norm, nb_iter=1, eps_iter=-.01)

  def test_eps_equals_zero(self):
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, self.attack, model_fn=self.model,
            x=self.x, eps=0, norm=norm, nb_iter=10, eps_iter=.01)
      else:
        self.assertClose(
            self.attack(
                model_fn=self.model, x=self.x, eps=0, norm=norm, nb_iter=10,
                eps_iter=.01),
            self.x)

  def test_eps_iter_equals_zero(self):
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, self.attack, model_fn=self.model, x=self.x,
            eps=.5, norm=norm, nb_iter=10, eps_iter=0)
      else:
        self.assertClose(
            self.attack(
                model_fn=self.model, x=self.x, eps=.5, norm=norm, nb_iter=10,
                eps_iter=0),
            self.x)

  def test_invalid_clips(self):
    clip_min = .5
    clip_max = -.5
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, self.attack, model_fn=self.model, x=self.x, eps=.1,
            norm=norm, clip_min=clip_min, clip_max=clip_max, nb_iter=10,
            eps_iter=.01)
      else:
        self.assertRaises(
            ValueError, self.attack, model_fn=self.model, x=self.x, eps=.1,
            norm=norm, clip_min=clip_min, clip_max=clip_max, nb_iter=10,
            eps_iter=.01)

  def test_adv_example_success_rate_linf(self):
    # use normalized_x to make sure the same eps gives uniformly high attack
    # success rate across randomized tests
    self.help_adv_examples_success_rate(
        norm=np.inf, **self.attack_param)

  def test_targeted_adv_example_success_rate_linf(self):
    self.help_targeted_adv_examples_success_rate(
        norm=np.inf, **self.attack_param)

  def test_adv_example_success_rate_l1(self):
    self.assertRaises(
        NotImplementedError, self.help_adv_examples_success_rate, norm=1,
        **self.attack_param)
    # TODO uncomment the actual test below after we have implemented the L1 attack
    # self.help_adv_examples_success_rate(
    #     norm=1, **self.attack_param)

  def test_targeted_adv_example_success_rate_l1(self):
    self.assertRaises(
        NotImplementedError, self.help_targeted_adv_examples_success_rate,
        norm=1, **self.attack_param)
    # TODO uncomment the actual test below after we have implemented the L1 attack
    # self.help_targeted_adv_examples_success_rate(
    #     norm=1, **self.attack_param)

  def test_adv_example_success_rate_l2(self):
    self.help_adv_examples_success_rate(
        norm=2, **self.attack_param)

  def test_targeted_adv_example_success_rate_l2(self):
    self.help_targeted_adv_examples_success_rate(
        norm=2, **self.attack_param)

  def test_do_not_reach_lp_boundary(self):
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, self.attack, model_fn=self.model,
            x=self.normalized_x, eps=.5, nb_iter=10, norm=norm, eps_iter=.01)
        continue
      else:
        x_adv = self.attack(
            model_fn=self.model, x=self.normalized_x, eps=.5, nb_iter=10,
            norm=norm, eps_iter=.01)

      if norm == np.inf:
        delta, _ = torch.abs(x_adv - self.normalized_x).max(dim=1)
      elif norm == 1:
        delta = torch.abs(x_adv - self.normalized_x).sum(dim=1)
      elif norm == 2:
        delta = torch.pow(x_adv - self.normalized_x, 2).sum(dim=1).pow(.5)
      diff = torch.max(.5 - delta)
      self.assertTrue(diff > .25)

  def test_attack_strength(self):
    x_adv = self.attack(
        model_fn=self.model, x=self.normalized_x, eps=1.,
        eps_iter=.05, norm=np.inf, clip_min=.5, clip_max=.7, nb_iter=5,
        sanity_checks=False)
    _, ori_label = self.model(self.normalized_x).max(1)
    _, adv_label = self.model(x_adv).max(1)
    adv_acc = (
        adv_label.eq(ori_label).sum().to(torch.float)
        / self.normalized_x.size(0))
    self.assertLess(adv_acc, .1)

  def test_eps(self):
    # test if the attack respects the norm constraint
    # NOTE clip_eta makes sure that at each step, adv_x respects the eps
    # norm constraint. Therefore, this is essentially a test on clip_eta,
    # which is implemented in a separate test_clip_eta
    raise SkipTest()

  def test_clip_eta(self):
    # NOTE: this has been tested with test_clip_eta in test_utils
    raise SkipTest()

  def test_clips(self):
    clip_min = -1.
    clip_max = 1.
    for norm in self.ord_list:
      if norm == 1:
        self.assertRaises(
            NotImplementedError, model_fn=self.model, x=self.normalized_x,
            eps=.3, eps_iter=.03, norm=norm, nb_iter=10, clip_min=clip_min,
            clip_max=clip_max)
        continue
      else:
        x_adv = self.attack(
            model_fn=self.model, x=self.normalized_x, eps=.3, eps_iter=.03,
            norm=norm, nb_iter=10, clip_min=clip_min, clip_max=clip_max)
      self.assertTrue(torch.all(x_adv <= clip_max))
      self.assertTrue(torch.all(x_adv >= clip_min))

  def test_attack_does_not_cache_graph_computation_for_nb_iter(self):
    # TODO not sure what the original test does in tests_tf/test_attacks
    pass

  def test_multiple_initial_random_step(self):
    _, ori_label = self.model(self.normalized_x).max(1)
    new_label_multi = ori_label.clone().detach()

    for _ in range(10):
      x_adv = self.attack(
          model_fn=self.model, x=self.normalized_x, eps=.5, eps_iter=.05,
          norm=np.inf, clip_min=.5, clip_max=.7, nb_iter=2, sanity_checks=False)
      _, new_label = self.model(x_adv).max(1)

      # examples for which we have not found adversarial examples
      i = ori_label.eq(new_label_multi)
      new_label_multi[i] = new_label[i]

    failed_attack = (
        ori_label.eq(new_label_multi).sum().to(torch.float)
        / self.normalized_x.size(0))
    self.assertLess(failed_attack, .5)


class TestSPSA(CommonAttackProperties):

  def setUp(self):
    super(TestSPSA, self).setUp()
    self.attack = spsa
    self.attack_param = {
        'eps': .5,
        'clip_min': -5,
        'clip_max': 5,
        'nb_iter': 50,
    }

  def test_invalid_input(self):
    x = torch.tensor([[-20., 30.]])
    self.assertRaises(AssertionError, self.attack, model_fn=self.model, x=x, eps=.1,
                      nb_iter=1, clip_min=-1., clip_max=1., sanity_checks=True)

  def test_invalid_eps(self):
    self.assertRaises(
        ValueError, self.attack, model_fn=self.model, x=self.x, eps=-.1, nb_iter=1)

  def test_eps_equals_zero(self):
    self.assertClose(
        self.attack(model_fn=self.model, x=self.x, eps=0, nb_iter=10),
        self.x)

  def test_invalid_clips(self):
    self.assertRaises(
        ValueError, self.attack, model_fn=self.model, x=self.x, eps=.1,
        clip_min=.5, clip_max=-.5, nb_iter=10)

  def test_adv_example_success_rate_linf(self):
    # use normalized_x to make sure the same eps gives uniformly high attack
    # success rate across randomized tests
    self.help_adv_examples_success_rate(**self.attack_param)

  def test_targeted_adv_example_success_rate_linf(self):
    self.help_targeted_adv_examples_success_rate(**self.attack_param)

  def test_adv_example_success_rate_l1(self):
    self.assertRaises(
        NotImplementedError, self.help_adv_examples_success_rate, norm=1,
        **self.attack_param)
    # TODO uncomment the actual test below after we have implemented the L1 attack
    # self.help_adv_examples_success_rate(
    #     norm=1, **self.attack_param)

  def test_targeted_adv_example_success_rate_l1(self):
    self.assertRaises(
        NotImplementedError, self.help_targeted_adv_examples_success_rate,
        norm=1, **self.attack_param)
    # TODO uncomment the actual test below after we have implemented the L1 attack
    # self.help_targeted_adv_examples_success_rate(
    #     norm=1, **self.attack_param)

  def test_adv_example_success_rate_l2(self):
    self.help_adv_examples_success_rate(
        norm=2, **self.attack_param)

  def test_targeted_adv_example_success_rate_l2(self):
    self.help_targeted_adv_examples_success_rate(
        norm=2, **self.attack_param)

  def test_attack_strength(self):
    x_adv = self.attack(
        model_fn=self.model, x=self.normalized_x, eps=1.,
        clip_min=.5, clip_max=.7, nb_iter=20,
        sanity_checks=False)
    _, ori_label = self.model(self.normalized_x).max(1)
    _, adv_label = self.model(x_adv).max(1)
    adv_acc = (
        adv_label.eq(ori_label).sum().to(torch.float)
        / self.normalized_x.size(0))
    self.assertLess(adv_acc, .1)

  def test_eps(self):
    x_adv = self.attack(
        model_fn=self.model, x=self.normalized_x, eps=.5, nb_iter=10)
    delta, _ = torch.abs(x_adv - self.normalized_x).max(dim=1)
    self.assertTrue(torch.all(delta <= .5 + 1e-6))

  def test_clips(self):
    clip_min = -1.
    clip_max = 1.
    x_adv = self.attack(
        model_fn=self.model, x=self.normalized_x, eps=.3,
        nb_iter=10, clip_min=clip_min, clip_max=clip_max)
    self.assertTrue(torch.all(x_adv <= clip_max))
    self.assertTrue(torch.all(x_adv >= clip_min))


class TestHopSkipJumpAttack(CommonAttackProperties):

  def setUp(self):
    super(TestHopSkipJumpAttack, self).setUp()
    self.attack = hop_skip_jump_attack

  def test_generate_np_untargeted_l2(self):
    x_val = torch.rand(50, 2)
    bapp_params = {
        'norm': 2,
        'stepsize_search': 'geometric_progression',
        'num_iterations': 10,
        'verbose': True,
    }
    x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

    _, ori_label = self.model(x_val).max(1)
    _, adv_label = self.model(x_adv).max(1)
    adv_acc = (
        adv_label.eq(ori_label).sum().to(torch.float)
        / x_val.size(0))

    self.assertLess(adv_acc, .1)

  def test_generate_untargeted_linf(self):
    x_val = torch.rand(50, 2)
    bapp_params = {
        'norm': np.inf,
        'stepsize_search': 'grid_search',
        'num_iterations': 10,
        'verbose': True,
    }
    x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

    _, ori_label = self.model(x_val).max(1)
    _, adv_label = self.model(x_adv).max(1)
    adv_acc = (
        adv_label.eq(ori_label).sum().to(torch.float)
        / x_val.size(0))

    self.assertLess(adv_acc, .1)

  def test_generate_np_targeted_linf(self):
    x_val = torch.rand(200, 2)

    _, ori_label = self.model(x_val).max(1)
    x_val_pos = x_val[ori_label == 1]
    x_val_neg = x_val[ori_label == 0]

    x_val_under_attack = torch.cat(
        (x_val_pos[:25], x_val_neg[:25]), dim=0)
    y_target = torch.cat(
        [torch.zeros(25, dtype=torch.int64), torch.ones(25, dtype=torch.int64)])
    image_target = torch.cat((x_val_neg[25:50], x_val_pos[25:50]), dim=0)

    bapp_params = {
        'norm': np.inf,
        'stepsize_search': 'geometric_progression',
        'num_iterations': 10,
        'verbose': True,
        'y_target': y_target,
        'image_target': image_target,
    }
    x_adv = self.attack(model_fn=self.model,
                        x=x_val_under_attack, **bapp_params)

    _, new_labs = self.model(x_adv).max(1)

    adv_acc = (
        new_labs.eq(y_target).sum().to(torch.float)
        / y_target.size(0))

    self.assertGreater(adv_acc, .9)

  def test_generate_targeted_l2(self):
    # Create data in numpy arrays.
    x_val = torch.rand(200, 2)

    _, ori_label = self.model(x_val).max(1)
    x_val_pos = x_val[ori_label == 1]
    x_val_neg = x_val[ori_label == 0]

    x_val_under_attack = torch.cat(
        (x_val_pos[:25], x_val_neg[:25]), dim=0)
    y_target = torch.cat(
        [torch.zeros(25, dtype=torch.int64), torch.ones(25, dtype=torch.int64)])
    image_target = torch.cat((x_val_neg[25:50], x_val_pos[25:50]), dim=0)

    # Create graph.
    bapp_params = {
        'norm': 'l2',
        'stepsize_search': 'grid_search',
        'num_iterations': 10,
        'verbose': True,
        'y_target': y_target,
        'image_target': image_target,
    }
    x_adv = self.attack(model_fn=self.model,
                        x=x_val_under_attack, **bapp_params)

    _, new_labs = self.model(x_adv).max(1)

    adv_acc = (
        new_labs.eq(y_target).sum().to(torch.float)
        / y_target.size(0))

    self.assertGreater(adv_acc, .9)


class TestCarliniWagnerL2(CommonAttackProperties):

  def setUp(self):
    super(TestCarliniWagnerL2, self).setUp()
    self.attack = carlini_wagner_l2

  def test_generate_untargeted_gives_adversarial_example(self):
    x_val = torch.rand(100, 2)
    bapp_params = {
        'max_iterations': 100,
        'binary_search_steps': 3,
        'initial_const': 1,
        'clip_min': -5,
        'clip_max': 5,
    }

    x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

    _, ori_label = self.model(x_val).max(1)
    _, adv_label = self.model(x_adv).max(1)
    adv_acc = (
        adv_label.eq(ori_label).sum().to(torch.float)
        / x_val.size(0))

    self.assertLess(adv_acc, .1)

  def test_generate_targeted_gives_adversarial_example(self):
    x_val = torch.rand(100, 2)
    feed_labs = torch.randint(0, 2, (100,))
    bapp_params = {
        'max_iterations': 100,
        'binary_search_steps': 3,
        'initial_const': 1,
        'clip_min': -5,
        'clip_max': 5,
        'y': feed_labs,
        'targeted': True
    }
    x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

    _, adv_label = self.model(x_adv).max(1)

    adv_acc = (
        adv_label.eq(feed_labs).sum().to(torch.float)
        / x_val.size(0))

    self.assertGreater(adv_acc, 0.9)

  def test_generate_gives_adversarial_example(self):
    x_val = torch.rand(100, 2)
    _, ori_label = self.model(x_val).max(1)
    bapp_params = {
        'max_iterations': 100,
        'binary_search_steps': 3,
        'initial_const': 1,
        'clip_min': -5,
        'clip_max': 5,
        'y': ori_label
    }
    x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

    _, adv_label = self.model(x_adv).max(1)
    adv_acc = (
        adv_label.eq(ori_label).sum().to(torch.float)
        / x_val.size(0))

    self.assertLess(adv_acc, .1)

  def test_generate_gives_clipped_adversarial_examples(self):
    x_val = torch.rand(100, 2)
    bapp_params = {
        'max_iterations': 10,
        'binary_search_steps': 1,
        'learning_rate': 1e-3,
        'initial_const': 1,
        'clip_min': -0.2,
        'clip_max': 0.3,
    }
    x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

    self.assertGreater(torch.min(x_adv),  -.201)
    self.assertLess(torch.max(x_adv), .301)

  def test_generate_high_confidence_targeted_examples(self):
    trivial_model = TrivialModel()

    for c in [0, 2.3]:
      x_val = torch.rand(10, 1) - .5
      feed_labs = torch.randint(0, 2, (10,))
      bapp_params = {
          'max_iterations': 100,
          'binary_search_steps': 2,
          'learning_rate': 1e-2,
          'initial_const': 1,
          'clip_min': -10,
          'clip_max': 10,
          'confidence': c,
          'y': feed_labs,
          'targeted': True
      }
      x_adv = self.attack(model_fn=trivial_model, x=x_val, **bapp_params)
      adv_logits = trivial_model(x_adv)
      adv_label = torch.argmax(adv_logits, 1)

      good_labs = adv_logits[torch.arange(10), feed_labs]
      bad_labs = adv_logits[torch.arange(10), 1 - feed_labs]

      adv_acc = (
          adv_label.eq(feed_labs).sum().to(torch.float)
          / x_val.size(0))

      diff = torch.min(good_labs - bad_labs).detach()
      self.assertClose(c, diff, atol=1e-1)
      self.assertGreater(adv_acc, .9)

  def test_generate_high_confidence_untargeted_examples(self):
    trivial_model = TrivialModel()

    for c in [0, 2.3]:
      x_val = torch.rand(10, 1) - .5
      _, ori_label = trivial_model(x_val).max(1)
      bapp_params = {
          'max_iterations': 100,
          'binary_search_steps': 2,
          'learning_rate': 1e-2,
          'initial_const': 1,
          'clip_min': -10,
          'clip_max': 10,
          'confidence': c
      }
      x_adv = self.attack(model_fn=trivial_model, x=x_val, **bapp_params)
      adv_logits = trivial_model(x_adv)
      adv_label = torch.argmax(adv_logits, 1)

      good_labs = adv_logits[torch.arange(10), 1 - ori_label]
      bad_labs = adv_logits[torch.arange(10), ori_label]

      adv_acc = (
          adv_label.eq(ori_label).sum().to(torch.float)
          / x_val.size(0))

      self.assertEqual(adv_acc, 0)
      diff = torch.min(good_labs - (bad_labs + c)).detach()
      self.assertClose(0, diff, atol=1e-1)
