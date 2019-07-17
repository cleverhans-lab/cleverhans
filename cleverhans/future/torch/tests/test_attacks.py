# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import copy
from nose.plugins.skip import SkipTest
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.future.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.future.torch.attacks.deepfool import deepfool

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


class SimpleImageModel(torch.nn.Module):

  def __init__(self):
    super(SimpleImageModel, self).__init__()
    self.w1 = torch.tensor([[1.5, .3], [-2, .3]])
    self.w2 = torch.tensor([[-2.4, 1.2], [.5, -2.3]])

  def forward(self, x):
    if len(x.size()) == 4:
      x = x[:, 0, 0]
    elif len(x.size()) == 3:
      x = x[None, 0, 0]
    x = torch.matmul(x, self.w1)
    x = torch.sigmoid(x)
    x = torch.matmul(x, self.w2)
    return x


class CommonAttackProperties(CleverHansTest):

  def setUp(self):
    super(CommonAttackProperties, self).setUp()
    self.model = SimpleModel()
    self.x = torch.randn(100, 2)
    self.normalized_x = torch.rand(100, 2) # truncated between [0, 1)
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
        'eps' : .5,
        'clip_min' : -5,
        'clip_max' : 5
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
        'eps' : .5,
        'clip_min' : -5,
        'clip_max' : 5,
        'eps_iter' : .05,
        'nb_iter' : 20,
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


class TestDeepFool(CommonAttackProperties):

  def setUp(self):
    super(TestDeepFool, self).setUp()
    self.attack = deepfool
    self.attack_param = {
        'eps' : .5,
        'clip_min' : -5,
        'clip_max' : 5,
        }

  def test_invalid_input(self):
    x = torch.tensor([[-2., 3.]])
    for norm in self.ord_list:
      self.assertRaises(
          AssertionError, self.attack, model_fn=self.model, x=x, eps=.1,
          norm=norm, clip_min=-1., clip_max=1., sanity_checks=True)

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

  def test_max_iter_equals_zero(self):
    for norm in self.ord_list:
      self.assertClose(
          self.attack(
              model_fn=self.model, x=self.x, eps=.5, norm=norm, max_iter=0),
          self.x)

  def test_invalid_clips(self):
    clip_min = .5
    clip_max = -.5
    for norm in self.ord_list:
      self.assertRaises(
          ValueError, self.attack, model_fn=self.model, x=self.x, eps=.1,
          norm=norm, clip_min=clip_min, clip_max=clip_max)

  def test_adv_example_success_rate_linf(self):
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

  def test_do_not_reach_lp_boundary(self):
    for norm in self.ord_list:
      x_adv = self.attack(
          model_fn=self.model, x=self.normalized_x, eps=.5, norm=norm)

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
        model_fn=self.model, x=self.normalized_x,
        clip_min=0., clip_max=1.,
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
      x_adv = self.attack(
          model_fn=self.model, x=self.normalized_x, eps=.3,
          norm=norm, clip_min=clip_min, clip_max=clip_max)
      self.assertTrue(torch.all(x_adv <= clip_max))
      self.assertTrue(torch.all(x_adv >= clip_min))

  def test_multiple_initial_random_step(self):
    _, ori_label = self.model(self.normalized_x).max(1)
    new_label_multi = ori_label.clone().detach()

    for _ in range(10):
      x_adv = self.attack(
          model_fn=self.model, x=self.normalized_x, eps=.5,
          norm=np.inf, clip_min=.5, clip_max=.7, sanity_checks=False)
      _, new_label = self.model(x_adv).max(1)

      # examples for which we have not found adversarial examples
      i = ori_label.eq(new_label_multi)
      new_label_multi[i] = new_label[i]

    failed_attack = (
        ori_label.eq(new_label_multi).sum().to(torch.float)
        / self.normalized_x.size(0))
    self.assertLess(failed_attack, .5)

  def test_matches_reference(self):
    model = SimpleImageModel()
    for image in self.x:
      image = image[None, None, :]
      _, _, _, _, pert_image = TestDeepFool.reference_deepfool(image, model, num_classes=2)
      self.assertClose(
          self.attack(model_fn=model, x=image[None])[0],
          pert_image)

  @staticmethod
  def reference_deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
    Reference implementation of DeepFool from original authors at
    https://github.com/LTS4/DeepFool.
     :param image: Image of size HxWx3
     :param net: network (input: images, output: values of activation **BEFORE** softmax).
     :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
     :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
     :param max_iter: maximum number of iterations for deepfool (default = 50)
     :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
      print("Using GPU")
      image = image.cuda()
      net = net.cuda()
    else:
      print("Using CPU")

    f_image = net.forward(Variable(
        image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

      pert = np.inf
      fs[0, I[0]].backward(retain_graph=True)
      grad_orig = x.grad.data.cpu().numpy().copy()

      for k in range(1, num_classes):
        zero_gradients(x)

        fs[0, I[k]].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()

        # set new w_k and new f_k
        w_k = cur_grad - grad_orig
        f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

        # determine which w_k to use
        if pert_k < pert:
            pert = pert_k
            w = w_k

      # compute r_i and r_tot
      # Added 1e-4 for numerical stability
      r_i = (pert + 1e-4) * w / np.linalg.norm(w)
      r_tot = np.float32(r_tot + r_i)

      if is_cuda:
        pert_image = image + (1 + overshoot) * \
            torch.from_numpy(r_tot).cuda()
      else:
        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

      x = Variable(pert_image, requires_grad=True)
      fs = net.forward(x)
      k_i = np.argmax(fs.data.cpu().numpy().flatten())

      loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image
