from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from distutils.version import LooseVersion

import unittest
import numpy as np
from collections import namedtuple

from cleverhans.utils import AccuracyReport
from cleverhans.devtools.checks import CleverHansTest
from attacks_multigpu import MadryEtAlMultiGPU

import sys
import os
sys.path.insert(0, os.path.abspath('../../tests_tf/'))
from test_attacks import TestMadryEtAl  # NOQA


class TestMadryEtAlMultiGPU(TestMadryEtAl):
    def setUp(self):
        super(TestMadryEtAlMultiGPU, self).setUp()
        self.attack_single_gpu = self.attack
        self.attack_multi_gpu = MadryEtAlMultiGPU(self.model, sess=self.sess,
                                                  ngpu=2)
        self.attack = self.attack_multi_gpu

    def test_single_vs_multi_gpu(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        def multi_attack(attack):
            flags = {'eps': 1.0, 'eps_iter': 0.05,
                     'clip_min': 0.5, 'clip_max': 0.7, 'nb_iter': 5}

            orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
            new_labs_multi = orig_labs.copy()
            # Generate multiple adversarial examples
            for i in range(10):
                x_adv = attack.generate_np(x_val, **flags)
                new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

                # Examples for which we have not found adversarial examples
                I = (orig_labs == new_labs_multi)
                new_labs_multi[I] = new_labs[I]

            return np.mean(orig_labs == new_labs_multi)

        acc_s = multi_attack(self.attack_single_gpu)
        acc_m = multi_attack(self.attack_multi_gpu)

        self.assertClose(acc_s, acc_m, atol=5e-3)


class TestRunMultiGPU(CleverHansTest):
    def test_run_multi_gpu_madryetal(self):
        np.random.seed(42)
        import tensorflow as tf
        tf.set_random_seed(42)

        from run_multigpu import run_trainer

        # Run the trainers on a dataset of reduced size
        flags = {'train_start': 0,
                 'train_end': 5000,
                 'test_start': 0,
                 'test_end': 333,
                 'nb_epochs': 2,
                 'testing': True}

        # Run the multi-gpu trainer for adversarial training
        flags.update({'batch_size': 128, 'learning_rate': 0.001,
                      'dataset': 'mnist', 'only_adv_train': False,
                      'eval_iters': 1, 'fast_tests': True,
                      'debug_graph': False,
                      'save_dir': None, 'save_steps': 10000,
                      'attack_nb_iter_train': 2, 'sync_step': None,
                      })

        flags.update({'adv_train': True,
                      'save': False,
                      'model_type': 'basic',
                      'ngpu': 1,
                      'attack_type_test': 'MadryEtAl_y',
                      'attack_type_train': 'MadryEtAl_y',
                      })
        HParams = namedtuple('HParams', flags.keys())

        hparams = HParams(**flags)
        report_dict = run_trainer(hparams)
        report_s = AccuracyReport()
        report_s.train_adv_train_clean_eval = report_dict['train']
        report_s.adv_train_clean_eval = report_dict['test']
        report_s.adv_train_adv_eval = report_dict['MadryEtAl_y']

        # Run the multi-gpu trainer for adversarial training using 2 gpus
        # trainer_multigpu by default sets `allow_soft_placement=True`
        flags.update({'ngpu': 2,
                      'attack_type_train': 'MadryEtAl_y_multigpu',
                      'sync_step': 1})
        hparams = HParams(**flags)
        report_dict = run_trainer(hparams)
        report_m = AccuracyReport()
        report_m.train_adv_train_clean_eval = report_dict['train']
        report_m.adv_train_clean_eval = report_dict['test']
        report_m.adv_train_adv_eval = report_dict['MadryEtAl_y']

        # Check that the tutorial is deterministic (seeded properly)
        if LooseVersion(tf.__version__) >= LooseVersion('1.1.0'):
            atol_fac = 3
        else:
            atol_fac = 2

        self.assertClose(report_s.train_adv_train_clean_eval,
                         report_m.train_adv_train_clean_eval,
                         atol=atol_fac * 1e-2)
        self.assertClose(report_s.adv_train_clean_eval,
                         report_m.adv_train_clean_eval,
                         atol=atol_fac * 5e-3)
        self.assertClose(report_s.adv_train_adv_eval,
                         report_m.adv_train_adv_eval,
                         atol=atol_fac * 5e-3)

    def test_run_single_gpu_fgsm(self):

        np.random.seed(42)
        import tensorflow as tf
        tf.set_random_seed(42)

        from cleverhans_tutorials import mnist_tutorial_tf
        from run_multigpu import run_trainer

        # Run the MNIST tutorial on a dataset of reduced size
        flags = {'train_start': 0,
                 'train_end': 5000,
                 'test_start': 0,
                 'test_end': 333,
                 'nb_epochs': 2,
                 'testing': True}
        report = mnist_tutorial_tf.mnist_tutorial(**flags)

        # Run the multi-gpu trainer for clean training
        flags.update({'batch_size': 128, 'learning_rate': 0.001,
                      'dataset': 'mnist', 'only_adv_train': False,
                      'eval_iters': 1, 'ngpu': 1, 'fast_tests': False,
                      'debug_graph': False, 'attack_type_train': None,
                      'save_dir': None, 'save_steps': 10000,
                      'attack_nb_iter_train': None, 'save': False,
                      'model_type': 'basic', 'attack_type_test': 'FGSM'})

        flags.update({'adv_train': False})
        HParams = namedtuple('HParams', flags.keys())

        hparams = HParams(**flags)
        report_dict = run_trainer(hparams)
        report_2 = AccuracyReport()
        report_2.train_clean_train_clean_eval = report_dict['train']
        report_2.clean_train_clean_eval = report_dict['test']
        report_2.clean_train_adv_eval = report_dict['FGSM']

        # Run the multi-gpu trainer for adversarial training
        flags.update({'adv_train': True,
                      'attack_type_train': 'FGSM',
                      })
        HParams = namedtuple('HParams', flags.keys())

        hparams = HParams(**flags)
        report_dict = run_trainer(hparams)
        report_2.train_adv_train_clean_eval = report_dict['train']
        report_2.adv_train_clean_eval = report_dict['test']
        report_2.adv_train_adv_eval = report_dict['FGSM']

        # Check that the tutorial is deterministic (seeded properly)
        if LooseVersion(tf.__version__) >= LooseVersion('1.1.0'):
            atol_fac = 4
        else:
            atol_fac = 2

        self.assertClose(report.train_clean_train_clean_eval,
                         report_2.train_clean_train_clean_eval,
                         atol=atol_fac * 5e-3)
        self.assertClose(report.clean_train_clean_eval,
                         report_2.clean_train_clean_eval,
                         atol=atol_fac * 5e-3)
        self.assertClose(report.clean_train_adv_eval,
                         report_2.clean_train_adv_eval,
                         atol=atol_fac * 5e-3)
        self.assertClose(report.train_adv_train_clean_eval,
                         report_2.train_adv_train_clean_eval,
                         atol=atol_fac * 2e-2)
        self.assertClose(report.adv_train_clean_eval,
                         report_2.adv_train_clean_eval,
                         atol=atol_fac * 2e-2)
        self.assertClose(report.adv_train_adv_eval,
                         report_2.adv_train_adv_eval,
                         atol=atol_fac * 2e-2)


if __name__ == '__main__':
    unittest.main()
