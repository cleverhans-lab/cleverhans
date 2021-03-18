# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from nose.plugins.skip import SkipTest
import torch

from cleverhans.devtools.checks import CleverHansTest
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent


class TrivialModel(torch.nn.Module):
    def __init__(self):
        super(TrivialModel, self).__init__()
        self.w1 = torch.tensor([[1.0, -1]])

    def forward(self, x, **kwargs):
        return torch.matmul(x, self.w1)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.w1 = torch.tensor([[1.5, 0.3], [-2, 0.3]])
        self.w2 = torch.tensor([[-2.4, 1.2], [0.5, -2.3]])

    def forward(self, x):
        x = torch.matmul(x, self.w1)
        x = torch.sigmoid(x)
        x = torch.matmul(x, self.w2)
        return x


class DummyModel(torch.nn.Module):
    def __init__(self, n_features):
        super(DummyModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_features, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 10),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)


class CommonAttackProperties(CleverHansTest):
    def setUp(self):
        super(CommonAttackProperties, self).setUp()
        self.model = SimpleModel()
        self.trivial_model = TrivialModel()
        self.x = torch.randn(100, 2)
        self.normalized_x = torch.rand(100, 2)  # truncated between [0, 1)
        self.trivial_x = torch.randn(100, 1)
        self.trivial_normalized_x = torch.rand(100, 1)  # truncated between [0, 1)
        self.y_target = torch.randint(low=0, high=2, size=(100,))
        self.ord_list = [1, 2, np.inf]

    def help_adv_examples_success_rate(self, model, x, rate=0.5, **kwargs):
        x_adv = self.attack(model_fn=model, x=x, **kwargs)
        _, ori_label = model(x).max(1)
        _, adv_label = model(x_adv).max(1)
        adv_acc = adv_label.eq(ori_label).sum().to(torch.float) / x.size(0)
        self.assertLess(adv_acc, rate)

    def help_targeted_adv_examples_success_rate(self, model, x, rate=0.7, **kwargs):
        x_adv = self.attack(
            model_fn=model, x=x, y=self.y_target, targeted=True, **kwargs
        )

        _, adv_label = model(x_adv).max(1)
        adv_success = adv_label.eq(self.y_target).sum().to(torch.float) / x.size(0)
        self.assertGreater(adv_success, rate)


class TestFastGradientMethod(CommonAttackProperties):
    def setUp(self):
        super(TestFastGradientMethod, self).setUp()
        self.attack = fast_gradient_method
        self.eps_list = [0, 0.1, 0.3, 1.0, 3]
        self.attack_param = {"eps": 0.5, "clip_min": -5, "clip_max": 5}

    def test_invalid_input(self):
        x = torch.tensor([[-2.0, 3.0]])
        for norm in self.ord_list:
            self.assertRaises(
                AssertionError,
                self.attack,
                model_fn=self.model,
                x=x,
                eps=0.1,
                norm=norm,
                clip_min=-1.0,
                clip_max=1.0,
                sanity_checks=True,
            )

    def test_invalid_eps(self):
        for norm in self.ord_list:
            self.assertRaises(
                ValueError,
                self.attack,
                model_fn=self.model,
                x=self.x,
                eps=-0.1,
                norm=norm,
            )

    def test_eps_equals_zero(self):
        for norm in self.ord_list:
            self.assertClose(
                self.attack(model_fn=self.model, x=self.x, eps=0, norm=norm), self.x
            )

    def test_eps(self):
        # test if the attack respects the norm constraint
        # NOTE this has been tested with the optimize_linear function in
        # test_utils, so duplicate tests are not needed here.
        # Although, if ever switch the engine of the FGM method to some
        # function other than optimize_linear. This test should be added.
        raise SkipTest()

    def test_clips(self):
        clip_min = -1.0
        clip_max = 1.0
        for norm in self.ord_list:
            x_adv = self.attack(
                model_fn=self.model,
                x=self.normalized_x,
                eps=0.3,
                norm=norm,
                clip_min=clip_min,
                clip_max=clip_max,
            )
            self.assertTrue(torch.all(x_adv <= clip_max))
            self.assertTrue(torch.all(x_adv >= clip_min))

    def test_invalid_clips(self):
        clip_min = 0.5
        clip_max = -0.5
        for norm in self.ord_list:
            self.assertRaises(
                ValueError,
                self.attack,
                model_fn=self.model,
                x=self.x,
                eps=0.1,
                norm=norm,
                clip_min=clip_min,
                clip_max=clip_max,
            )

    def test_adv_example_success_rate_linf(self):
        # use normalized_x to make sure the same eps gives uniformly high attack
        # success rate across randomized tests
        self.help_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=np.inf, **self.attack_param
        )

    def test_targeted_adv_example_success_rate_linf(self):
        self.help_targeted_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=np.inf, **self.attack_param
        )

    def test_adv_example_success_rate_l1(self):
        self.help_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=1, **self.attack_param
        )

    def test_targeted_adv_example_success_rate_l1(self):
        self.help_targeted_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=1, **self.attack_param
        )

    def test_adv_example_success_rate_l2(self):
        self.help_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=2, **self.attack_param
        )

    def test_targeted_adv_example_success_rate_l2(self):
        self.help_targeted_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=2, **self.attack_param
        )


class TestProjectedGradientMethod(CommonAttackProperties):
    def setUp(self):
        super(TestProjectedGradientMethod, self).setUp()
        self.attack = projected_gradient_descent
        self.attack_param = {
            "eps": 0.5,
            "clip_min": -5,
            "clip_max": 5,
            "eps_iter": 0.05,
            "nb_iter": 20,
        }

    def test_invalid_input(self):
        x = torch.tensor([[-2.0, 3.0]])
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    self.attack,
                    model_fn=self.model,
                    x=x,
                    eps=0.1,
                    nb_iter=1,
                    eps_iter=0.05,
                    norm=norm,
                    clip_min=-1.0,
                    clip_max=1.0,
                    sanity_checks=True,
                )
            else:
                self.assertRaises(
                    AssertionError,
                    self.attack,
                    model_fn=self.model,
                    x=x,
                    eps=0.1,
                    nb_iter=1,
                    eps_iter=0.05,
                    norm=norm,
                    clip_min=-1.0,
                    clip_max=1.0,
                    sanity_checks=True,
                )

    def test_invalid_eps(self):
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=-0.1,
                    norm=norm,
                    nb_iter=1,
                    eps_iter=0.01,
                )
            else:
                self.assertRaises(
                    ValueError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=-0.1,
                    norm=norm,
                    nb_iter=1,
                    eps_iter=0.01,
                )

    def test_invalid_eps_iter(self):
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=0.1,
                    norm=norm,
                    nb_iter=1,
                    eps_iter=-0.01,
                )
            else:
                self.assertRaises(
                    ValueError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=0.1,
                    norm=norm,
                    nb_iter=1,
                    eps_iter=-0.01,
                )

    def test_eps_equals_zero(self):
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=0,
                    norm=norm,
                    nb_iter=10,
                    eps_iter=0.01,
                )
            else:
                self.assertClose(
                    self.attack(
                        model_fn=self.model,
                        x=self.x,
                        eps=0,
                        norm=norm,
                        nb_iter=10,
                        eps_iter=0.01,
                    ),
                    self.x,
                )

    def test_eps_iter_equals_zero(self):
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=0.5,
                    norm=norm,
                    nb_iter=10,
                    eps_iter=0,
                )
            else:
                self.assertClose(
                    self.attack(
                        model_fn=self.model,
                        x=self.x,
                        eps=0.5,
                        norm=norm,
                        nb_iter=10,
                        eps_iter=0,
                    ),
                    self.x,
                )

    def test_invalid_clips(self):
        clip_min = 0.5
        clip_max = -0.5
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=0.1,
                    norm=norm,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    nb_iter=10,
                    eps_iter=0.01,
                )
            else:
                self.assertRaises(
                    ValueError,
                    self.attack,
                    model_fn=self.model,
                    x=self.x,
                    eps=0.1,
                    norm=norm,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    nb_iter=10,
                    eps_iter=0.01,
                )

    def test_adv_example_success_rate_linf(self):
        # use normalized_x to make sure the same eps gives uniformly high attack
        # success rate across randomized tests
        self.help_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=np.inf, **self.attack_param
        )

    def test_targeted_adv_example_success_rate_linf(self):
        self.help_targeted_adv_examples_success_rate(
            x=self.normalized_x, model=self.model, norm=np.inf, **self.attack_param
        )

    def test_adv_example_success_rate_l1(self):
        self.assertRaises(
            NotImplementedError,
            self.help_adv_examples_success_rate,
            model=self.model,
            x=self.normalized_x,
            norm=1,
            **self.attack_param
        )
        # TODO uncomment the actual test below after we have implemented the L1 attack
        # self.help_adv_examples_success_rate(x=self.normalized_x,
        #     model=self.model, norm=1, **self.attack_param)

    def test_targeted_adv_example_success_rate_l1(self):
        self.assertRaises(
            NotImplementedError,
            self.help_targeted_adv_examples_success_rate,
            x=self.normalized_x,
            model=self.model,
            norm=1,
            **self.attack_param
        )
        # TODO uncomment the actual test below after we have implemented the L1 attack
        # self.help_targeted_adv_examples_success_rate(x=self.normalized_x,
        #     model=self.model, norm=1, **self.attack_param)

    def test_adv_example_success_rate_l2(self):
        self.help_adv_examples_success_rate(
            model=self.model, x=self.normalized_x, norm=2, **self.attack_param
        )

    def test_targeted_adv_example_success_rate_l2(self):
        self.help_targeted_adv_examples_success_rate(
            model=self.model, x=self.normalized_x, norm=2, **self.attack_param
        )

    def test_do_not_reach_lp_boundary(self):
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    self.attack,
                    model_fn=self.model,
                    x=self.normalized_x,
                    eps=0.5,
                    nb_iter=10,
                    norm=norm,
                    eps_iter=0.01,
                )
                continue
            else:
                x_adv = self.attack(
                    model_fn=self.model,
                    x=self.normalized_x,
                    eps=0.5,
                    nb_iter=10,
                    norm=norm,
                    eps_iter=0.01,
                )

            if norm == np.inf:
                delta, _ = torch.abs(x_adv - self.normalized_x).max(dim=1)
            elif norm == 1:
                delta = torch.abs(x_adv - self.normalized_x).sum(dim=1)
            elif norm == 2:
                delta = torch.pow(x_adv - self.normalized_x, 2).sum(dim=1).pow(0.5)
            diff = torch.max(0.5 - delta)
            self.assertTrue(diff > 0.25)

    def test_attack_strength(self):
        x_adv = self.attack(
            model_fn=self.model,
            x=self.normalized_x,
            eps=1.0,
            eps_iter=0.05,
            norm=np.inf,
            clip_min=0.5,
            clip_max=0.7,
            nb_iter=5,
            sanity_checks=False,
        )
        _, ori_label = self.model(self.normalized_x).max(1)
        _, adv_label = self.model(x_adv).max(1)
        adv_acc = adv_label.eq(ori_label).sum().to(
            torch.float
        ) / self.normalized_x.size(0)
        self.assertLess(adv_acc, 0.1)

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
        clip_min = -1.0
        clip_max = 1.0
        for norm in self.ord_list:
            if norm == 1:
                self.assertRaises(
                    NotImplementedError,
                    model_fn=self.model,
                    x=self.normalized_x,
                    eps=0.3,
                    eps_iter=0.03,
                    norm=norm,
                    nb_iter=10,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
                continue
            else:
                x_adv = self.attack(
                    model_fn=self.model,
                    x=self.normalized_x,
                    eps=0.3,
                    eps_iter=0.03,
                    norm=norm,
                    nb_iter=10,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
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
                model_fn=self.model,
                x=self.normalized_x,
                eps=0.5,
                eps_iter=0.05,
                norm=np.inf,
                clip_min=0.5,
                clip_max=0.7,
                nb_iter=2,
                sanity_checks=False,
            )
            _, new_label = self.model(x_adv).max(1)

            # examples for which we have not found adversarial examples
            i = ori_label.eq(new_label_multi)
            new_label_multi[i] = new_label[i]

        failed_attack = ori_label.eq(new_label_multi).sum().to(
            torch.float
        ) / self.normalized_x.size(0)
        self.assertLess(failed_attack, 0.5)


class TestCarliniWagnerL2(CommonAttackProperties):
    def setUp(self):
        super(TestCarliniWagnerL2, self).setUp()
        self.attack = carlini_wagner_l2
        self.attack_param = {
            "n_classes": 2,
            "max_iterations": 100,
            "binary_search_steps": 3,
            "initial_const": 1,
        }

    def test_adv_example_success_rate(self):
        self.help_adv_examples_success_rate(
            model=self.model,
            x=self.normalized_x,
            rate=0.1,
            clip_min=-5,
            clip_max=5,
            **self.attack_param
        )

    def test_targeted_adv_example_success_rate(self):
        self.help_targeted_adv_examples_success_rate(
            model=self.model,
            x=self.normalized_x,
            rate=0.9,
            clip_min=-5,
            clip_max=5,
            **self.attack_param
        )

    def test_adv_examples_clipped_successfully(self):
        x_adv = self.attack(
            model_fn=self.model,
            x=self.normalized_x,
            clip_min=-0.2,
            clip_max=0.3,
            **self.attack_param
        )
        self.assertGreater(torch.min(x_adv), -0.201)
        self.assertLess(torch.max(x_adv), 0.301)

    def test_high_confidence_adv_example(self):
        from copy import copy

        attack_param_copy = copy(self.attack_param)
        attack_param_copy["binary_search_steps"] = 2

        x = self.trivial_normalized_x - 0.5
        _, y = self.trivial_model(x).max(1)

        for confidence in [0, 2.3]:
            x_adv = self.attack(
                model_fn=self.trivial_model,
                x=x,
                lr=1e-2,
                clip_min=-10,
                clip_max=10,
                confidence=confidence,
                **attack_param_copy
            )
            logits = self.trivial_model(x_adv)
            target = logits[range(len(logits)), 1 - y]
            other = logits[range(len(logits)), y]
            self.assertClose(confidence, torch.min(target - other).detach(), atol=1e-1)
            self.assertTrue(
                torch.argmax(logits, 1).eq(y).sum().to(torch.float) / len(logits) == 0
            )

    def test_high_confidence_targeted_adv_example(self):
        from copy import copy

        attack_param_copy = copy(self.attack_param)
        attack_param_copy["binary_search_steps"] = 2

        for confidence in [0, 2.3]:
            x_adv = self.attack(
                model_fn=self.trivial_model,
                x=self.trivial_normalized_x - 0.5,
                lr=1e-2,
                clip_min=-10,
                clip_max=10,
                targeted=True,
                y=self.y_target,
                confidence=confidence,
                **attack_param_copy
            )
            logits = self.trivial_model(x_adv)
            target = logits[range(len(logits)), self.y_target]
            other = logits[range(len(logits)), 1 - self.y_target]
            self.assertClose(confidence, torch.min(target - other).detach(), atol=1e-1)
            self.assertGreater(
                torch.argmax(logits, 1).eq(self.y_target).sum().to(torch.float)
                / len(logits),
                0.9,
            )


class TestSPSA(CommonAttackProperties):
    def setUp(self):
        super(TestSPSA, self).setUp()
        self.attack = spsa
        self.attack_param = {
            "eps": 0.5,
            "clip_min": -5,
            "clip_max": 5,
            "nb_iter": 50,
            "model": self.model,
            "x": self.normalized_x,
        }

    def test_invalid_input(self):
        x = torch.tensor([[-20.0, 30.0]])
        self.assertRaises(
            AssertionError,
            self.attack,
            model_fn=self.model,
            x=x,
            eps=0.1,
            nb_iter=1,
            clip_min=-1.0,
            clip_max=1.0,
            sanity_checks=True,
        )

    def test_invalid_eps(self):
        self.assertRaises(
            ValueError, self.attack, model_fn=self.model, x=self.x, eps=-0.1, nb_iter=1
        )

    def test_eps_equals_zero(self):
        self.assertClose(
            self.attack(model_fn=self.model, x=self.x, eps=0, nb_iter=10), self.x
        )

    def test_invalid_clips(self):
        self.assertRaises(
            ValueError,
            self.attack,
            model_fn=self.model,
            x=self.x,
            eps=0.1,
            clip_min=0.5,
            clip_max=-0.5,
            nb_iter=10,
        )

    def test_adv_example_success_rate_linf(self):
        # use normalized_x to make sure the same eps gives uniformly high attack
        # success rate across randomized tests
        self.help_adv_examples_success_rate(**self.attack_param)

    def test_targeted_adv_example_success_rate_linf(self):
        self.help_targeted_adv_examples_success_rate(**self.attack_param)

    def test_adv_example_success_rate_l1(self):
        self.assertRaises(
            NotImplementedError,
            self.help_adv_examples_success_rate,
            norm=1,
            **self.attack_param
        )
        # TODO uncomment the actual test below after we have implemented the L1 attack
        # self.help_adv_examples_success_rate(
        #     norm=1, **self.attack_param)

    def test_targeted_adv_example_success_rate_l1(self):
        self.assertRaises(
            NotImplementedError,
            self.help_targeted_adv_examples_success_rate,
            norm=1,
            **self.attack_param
        )
        # TODO uncomment the actual test below after we have implemented the L1 attack
        # self.help_targeted_adv_examples_success_rate(
        #     norm=1, **self.attack_param)

    def test_adv_example_success_rate_l2(self):
        self.help_adv_examples_success_rate(norm=2, **self.attack_param)

    def test_targeted_adv_example_success_rate_l2(self):
        self.help_targeted_adv_examples_success_rate(norm=2, **self.attack_param)

    def test_attack_strength(self):
        x_adv = self.attack(
            model_fn=self.model,
            x=self.normalized_x,
            eps=1.0,
            clip_min=0.5,
            clip_max=0.7,
            nb_iter=20,
            sanity_checks=False,
        )
        _, ori_label = self.model(self.normalized_x).max(1)
        _, adv_label = self.model(x_adv).max(1)
        adv_acc = adv_label.eq(ori_label).sum().to(
            torch.float
        ) / self.normalized_x.size(0)
        self.assertLess(adv_acc, 0.1)

    def test_eps(self):
        x_adv = self.attack(
            model_fn=self.model, x=self.normalized_x, eps=0.5, nb_iter=10
        )
        delta, _ = torch.abs(x_adv - self.normalized_x).max(dim=1)
        self.assertTrue(torch.all(delta <= 0.5 + 1e-6))

    def test_clips(self):
        clip_min = -1.0
        clip_max = 1.0
        x_adv = self.attack(
            model_fn=self.model,
            x=self.normalized_x,
            eps=0.3,
            nb_iter=10,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.assertTrue(torch.all(x_adv <= clip_max))
        self.assertTrue(torch.all(x_adv >= clip_min))


class TestHopSkipJumpAttack(CommonAttackProperties):
    def setUp(self):
        super(TestHopSkipJumpAttack, self).setUp()
        self.attack = hop_skip_jump_attack

    def test_generate_np_untargeted_l2(self):
        x_val = torch.rand(50, 2)
        bapp_params = {
            "norm": 2,
            "stepsize_search": "geometric_progression",
            "num_iterations": 10,
            "verbose": True,
        }
        x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

        _, ori_label = self.model(x_val).max(1)
        _, adv_label = self.model(x_adv).max(1)
        adv_acc = adv_label.eq(ori_label).sum().to(torch.float) / x_val.size(0)

        self.assertLess(adv_acc, 0.1)

    def test_generate_untargeted_linf(self):
        x_val = torch.rand(50, 2)
        bapp_params = {
            "norm": np.inf,
            "stepsize_search": "grid_search",
            "num_iterations": 10,
            "verbose": True,
        }
        x_adv = self.attack(model_fn=self.model, x=x_val, **bapp_params)

        _, ori_label = self.model(x_val).max(1)
        _, adv_label = self.model(x_adv).max(1)
        adv_acc = adv_label.eq(ori_label).sum().to(torch.float) / x_val.size(0)

        self.assertLess(adv_acc, 0.1)

    def test_generate_np_targeted_linf(self):
        x_val = torch.rand(200, 2)

        _, ori_label = self.model(x_val).max(1)
        x_val_pos = x_val[ori_label == 1]
        x_val_neg = x_val[ori_label == 0]

        x_val_under_attack = torch.cat((x_val_pos[:25], x_val_neg[:25]), dim=0)
        y_target = torch.cat(
            [torch.zeros(25, dtype=torch.int64), torch.ones(25, dtype=torch.int64)]
        )
        image_target = torch.cat((x_val_neg[25:50], x_val_pos[25:50]), dim=0)

        bapp_params = {
            "norm": np.inf,
            "stepsize_search": "geometric_progression",
            "num_iterations": 10,
            "verbose": True,
            "y_target": y_target,
            "image_target": image_target,
        }
        x_adv = self.attack(model_fn=self.model, x=x_val_under_attack, **bapp_params)

        _, new_labs = self.model(x_adv).max(1)

        adv_acc = new_labs.eq(y_target).sum().to(torch.float) / y_target.size(0)

        self.assertGreater(adv_acc, 0.9)

    def test_generate_targeted_l2(self):
        # Create data in numpy arrays.
        x_val = torch.rand(200, 2)

        _, ori_label = self.model(x_val).max(1)
        x_val_pos = x_val[ori_label == 1]
        x_val_neg = x_val[ori_label == 0]

        x_val_under_attack = torch.cat((x_val_pos[:25], x_val_neg[:25]), dim=0)
        y_target = torch.cat(
            [torch.zeros(25, dtype=torch.int64), torch.ones(25, dtype=torch.int64)]
        )
        image_target = torch.cat((x_val_neg[25:50], x_val_pos[25:50]), dim=0)

        # Create graph.
        bapp_params = {
            "norm": "l2",
            "stepsize_search": "grid_search",
            "num_iterations": 10,
            "verbose": True,
            "y_target": y_target,
            "image_target": image_target,
        }
        x_adv = self.attack(model_fn=self.model, x=x_val_under_attack, **bapp_params)

        _, new_labs = self.model(x_adv).max(1)

        adv_acc = new_labs.eq(y_target).sum().to(torch.float) / y_target.size(0)

        self.assertGreater(adv_acc, 0.9)


class TestSparseL1Descent(CommonAttackProperties):
    def setUp(self):
        super(TestSparseL1Descent, self).setUp()
        self.attack = sparse_l1_descent

    def generate_adversarial_examples(self, **kwargs):
        x_adv = self.attack(model_fn=self.model, x=self.normalized_x, **kwargs)
        _, ori_label = self.model(self.normalized_x).max(1)
        _, adv_label = self.model(x_adv).max(1)
        adv_acc = adv_label.eq(ori_label).sum().to(
            torch.float
        ) / self.normalized_x.size(0)

        delta = torch.sum(torch.abs(x_adv - self.normalized_x), dim=1)
        return x_adv, delta, adv_acc

    def generate_targeted_adversarial_examples(self, **kwargs):
        y_target = torch.randint(low=0, high=2, size=(self.normalized_x.size(0),))
        x_adv = self.attack(
            model_fn=self.model,
            x=self.normalized_x,
            y=y_target,
            targeted=True,
            **kwargs
        )

        _, adv_label = self.model(x_adv).max(1)
        adv_success = adv_label.eq(y_target).sum().to(
            torch.float
        ) / self.normalized_x.size(0)

        delta = torch.sum(torch.abs(x_adv - self.normalized_x), dim=1)
        return x_adv, delta, adv_success

    def test_invalid_input(self):
        x_val = -torch.ones((2, 2))
        with self.assertRaises(AssertionError):
            self.attack(self.model, x_val, eps=10.0, clip_min=0.0, clip_max=1.0)

    def test_gives_adversarial_example(self):
        _, delta, adv_acc = self.generate_adversarial_examples(
            eps=2, clip_min=-5, clip_max=5
        )
        self.assertLess(adv_acc, 0.5)
        self.assertLess(torch.max(torch.abs(delta - 2)), 1e-3)

    def test_targeted_gives_adversarial_example(self):
        _, delta, adv_acc = self.generate_targeted_adversarial_examples(
            eps=10, clip_min=-5, clip_max=5
        )
        self.assertGreater(adv_acc, 0.7)
        self.assertLessEqual(torch.max(delta), 10.001)

    def test_can_be_called_with_different_eps(self):
        for eps in [10, 20, 30, 40]:
            _, delta, _ = self.generate_adversarial_examples(
                eps=eps, clip_min=-5, clip_max=5
            )
            self.assertLessEqual(torch.max(delta), eps + 1e-4)

    def test_clip_works_as_expected(self):
        x_adv, _, _ = self.generate_adversarial_examples(
            eps=10,
            nb_iter=20,
            rand_init=True,
            clip_min=-0.2,
            clip_max=0.1,
            sanity_checks=False,
        )

        self.assertClose(torch.min(x_adv), -0.2)
        self.assertClose(torch.max(x_adv), 0.1)

    def test_do_not_reach_lp_boundary(self):
        """
        Make sure that iterative attack don't reach boundary of Lp
        neighbourhood if nb_iter * eps_iter is relatively small compared to
        epsilon.
        """
        _, delta, _ = self.generate_adversarial_examples(
            eps=0.5, clip_min=-5, clip_max=5, nb_iter=10, eps_iter=0.01
        )
        self.assertTrue(torch.max(0.5 - delta) > 0.25)

    def test_generate_np_gives_clipped_adversarial_examples(self):
        x_adv, _, _ = self.generate_adversarial_examples(
            eps=1.0,
            eps_iter=0.1,
            nb_iter=5,
            clip_min=-0.2,
            clip_max=0.3,
            sanity_checks=False,
        )

        self.assertLess(-0.201, torch.min(x_adv))
        self.assertLess(torch.max(x_adv), 0.301)

    def test_clip_eta(self):
        _, delta, _ = self.generate_adversarial_examples(
            eps=1, clip_min=-5, clip_max=5, nb_iter=5, eps_iter=0.1
        )

        # this projection is less numerically stable so give it some slack
        self.assertLessEqual(torch.max(delta), 1.0 + 1e-6)

    def test_attack_strength(self):
        # sanity checks turned off because this test initializes outside
        # the valid range.
        _, _, adv_acc = self.generate_adversarial_examples(
            eps=10,
            rand_init=True,
            clip_min=0.5,
            clip_max=0.7,
            nb_iter=10,
            sanity_checks=False,
        )

        self.assertLess(adv_acc, 0.4)

    def test_grad_clip(self):
        """
        With clipped gradients, we achieve
        np.mean(orig_labels == new_labels) == 0.0
        """

        # sanity checks turned off because this test initializes outside
        # the valid range.
        _, _, adv_acc = self.generate_adversarial_examples(
            eps=10,
            rand_init=True,
            clip_grad=True,
            clip_min=0.5,
            clip_max=0.7,
            nb_iter=10,
            sanity_checks=False,
        )
        self.assertLess(adv_acc, 0.1)

    def test_sparsity(self):
        # use a model with larger input dimensionality for this test.
        model_fn = DummyModel(1000)
        x_val = torch.rand(100, 1000)

        for q in [1, 9, 25.8, 50, 75.4, 90.2, 99, 99.9]:
            x_adv = self.attack(
                model_fn,
                x_val,
                eps=5.0,
                grad_sparsity=q,
                nb_iter=1,
                sanity_checks=False,
            )

            numzero = torch.sum(x_adv - x_val == 0, dim=-1).float()
            self.assertAlmostEqual(q * 1000.0 / 100.0, torch.mean(numzero), delta=1)

    def test_grad_sparsity_checks(self):
        # test that the attacks allows `grad_sparsity` to be specified as a scalar
        # in (0, 100) or as a vector.

        # scalar values out of range
        with self.assertRaises(ValueError):
            self.generate_adversarial_examples(grad_sparsity=0)

        with self.assertRaises(ValueError):
            self.generate_adversarial_examples(grad_sparsity=100)

        # sparsity as 2D array should fail
        with self.assertRaises(ValueError):
            gs = torch.empty(100, 2).uniform_(90, 99)
            self.generate_adversarial_examples(sanity_checks=False, grad_sparsity=gs)

        # sparsity as 1D array should succeed
        gs = torch.empty(100).uniform_(90, 99)
        self.generate_adversarial_examples(sanity_checks=False, grad_sparsity=gs)

        # sparsity vector of wrong size should fail
        with self.assertRaises(ValueError) as context:
            gs = torch.empty(101).uniform_(90, 99)
            self.generate_adversarial_examples(sanity_checks=False, grad_sparsity=gs)
