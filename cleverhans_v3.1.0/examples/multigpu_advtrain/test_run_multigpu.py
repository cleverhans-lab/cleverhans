# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import unittest

import numpy as np
import tensorflow as tf

from cleverhans.utils import AccuracyReport
from cleverhans.devtools.checks import CleverHansTest
from run_multigpu import run_trainer


class TestRunMultiGPU(CleverHansTest):
  def helper_run_multi_gpu_madryetal(self, extra_flags=None):
    """
    Compare the single GPU performance to multiGPU performance.
    """
    # Run the trainers on a dataset of reduced size
    flags = {'train_start': 0,
             'train_end': 5000,
             'test_start': 0,
             'test_end': 333,
             'nb_epochs': 5,
             'testing': True}

    # Run the multi-gpu trainer for adversarial training
    flags.update({'batch_size': 128, 'adam_lrn': 0.001,
                  'dataset': 'mnist', 'only_adv_train': False,
                  'eval_iters': 1, 'fast_tests': True,
                  'save_dir': None, 'save_steps': 10000,
                  'attack_nb_iter_train': 10, 'sync_step': None,
                  'adv_train': True,
                  'save': False,
                  'model_type': 'basic',
                  'attack_type_test': 'MadryEtAl_y'})
    if extra_flags is not None:
      flags.update(extra_flags)

    # Run the multi-gpu trainer for adversarial training using 2 gpus
    # trainer_multigpu by default sets `allow_soft_placement=True`
    flags.update({'ngpu': 2,
                  'attack_type_train': 'MadryEtAl_y_multigpu',
                  'sync_step': 1})
    HParams = namedtuple('HParams', flags.keys())

    hparams = HParams(**flags)
    np.random.seed(42)
    tf.set_random_seed(42)
    with tf.variable_scope(None, 'runner'):
      report_dict = run_trainer(hparams)
    report_m = AccuracyReport()
    report_m.train_adv_train_clean_eval = report_dict['train']
    report_m.adv_train_clean_eval = report_dict['test']
    report_m.adv_train_adv_eval = report_dict['MadryEtAl_y']

    flags.update({'ngpu': 1, 'attack_type_train': 'MadryEtAl_y'})
    hparams = HParams(**flags)
    np.random.seed(42)
    tf.set_random_seed(42)
    with tf.variable_scope(None, 'runner'):
      report_dict = run_trainer(hparams)
    report_s = AccuracyReport()
    report_s.train_adv_train_clean_eval = report_dict['train']
    report_s.adv_train_clean_eval = report_dict['test']
    report_s.adv_train_adv_eval = report_dict['MadryEtAl_y']

    self.assertClose(report_s.train_adv_train_clean_eval,
                     report_m.train_adv_train_clean_eval,
                     atol=5e-2)
    self.assertClose(report_s.adv_train_clean_eval,
                     report_m.adv_train_clean_eval,
                     atol=2e-2)
    self.assertClose(report_s.adv_train_adv_eval,
                     report_m.adv_train_adv_eval,
                     atol=5e-2)

  def test_run_single_gpu_fgsm(self):
    """
    Test the basic single GPU performance by comparing to the FGSM
    tutorial.
    """
    from cleverhans_tutorials import mnist_tutorial_tf

    # Run the MNIST tutorial on a dataset of reduced size
    flags = {'train_start': 0,
             'train_end': 5000,
             'test_start': 0,
             'test_end': 333,
             'nb_epochs': 5,
             'testing': True}
    report = mnist_tutorial_tf.mnist_tutorial(**flags)

    # Run the multi-gpu trainer for clean training
    flags.update({'batch_size': 128, 'adam_lrn': 0.001,
                  'dataset': 'mnist', 'only_adv_train': False,
                  'eval_iters': 1, 'ngpu': 1, 'fast_tests': False,
                  'attack_type_train': '',
                  'save_dir': None, 'save_steps': 10000,
                  'attack_nb_iter_train': None, 'save': False,
                  'model_type': 'basic', 'attack_type_test': 'FGSM'})

    flags.update({'adv_train': False})
    HParams = namedtuple('HParams', flags.keys())

    hparams = HParams(**flags)
    np.random.seed(42)
    tf.set_random_seed(42)
    with tf.variable_scope(None, 'runner'):
      report_dict = run_trainer(hparams)
    report_2 = AccuracyReport()
    report_2.train_clean_train_clean_eval = report_dict['train']
    report_2.clean_train_clean_eval = report_dict['test']
    report_2.clean_train_adv_eval = report_dict['FGSM']

    # Run the multi-gpu trainer for adversarial training
    flags.update({'adv_train': True, 'attack_type_train': 'FGSM'})
    HParams = namedtuple('HParams', flags.keys())

    hparams = HParams(**flags)
    np.random.seed(42)
    tf.set_random_seed(42)
    with tf.variable_scope(None, 'runner'):
      report_dict = run_trainer(hparams)
    report_2.train_adv_train_clean_eval = report_dict['train']
    report_2.adv_train_clean_eval = report_dict['test']
    report_2.adv_train_adv_eval = report_dict['FGSM']

    self.assertClose(report.train_clean_train_clean_eval,
                     report_2.train_clean_train_clean_eval,
                     atol=5e-2)
    self.assertClose(report.clean_train_clean_eval,
                     report_2.clean_train_clean_eval,
                     atol=2e-2)
    self.assertClose(report.clean_train_adv_eval,
                     report_2.clean_train_adv_eval,
                     atol=5e-2)
    self.assertClose(report.train_adv_train_clean_eval,
                     report_2.train_adv_train_clean_eval,
                     atol=1e-1)
    self.assertClose(report.adv_train_clean_eval,
                     report_2.adv_train_clean_eval,
                     atol=2e-2)
    self.assertClose(report.adv_train_adv_eval,
                     report_2.adv_train_adv_eval,
                     atol=1e-1)

  def test_run_multi_gpu_madryetal(self):
    self.helper_run_multi_gpu_madryetal()

  def test_run_multi_gpu_naive(self):
    self.helper_run_multi_gpu_madryetal({'adv_train': False})


if __name__ == '__main__':
  unittest.main()
