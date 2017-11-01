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
from cleverhans.attacks import MadryEtAl


class TestMadryEtAl(CleverHansTest):
    def setUp(self):
        super(TestMadryEtAl, self).setUp()
        import tensorflow as tf

        # The world's simplest neural network
        def my_model(x):
            W1 = tf.constant([[1.5, .3], [-2, 0.3]], dtype=tf.float32)
            h1 = tf.nn.sigmoid(tf.matmul(x, W1))
            W2 = tf.constant([[-2.4, 1.2], [0.5, -2.3]], dtype=tf.float32)
            res = tf.matmul(h1, W2)
            return res

        self.sess = tf.Session()
        self.model = my_model
        self.attack = MadryEtAl(self.model, sess=self.sess)

    def test_attack_strength(self):
        """
        If clipping is not done at each iteration (not using clip_min and
        clip_max), this attack fails by
        np.mean(orig_labels == new_labels) == .5
        """
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.05,
                                        clip_min=0.5, clip_max=0.7,
                                        nb_iter=5)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)
        print(np.mean(orig_labs == new_labs))
        self.assertTrue(np.mean(orig_labs == new_labs) < 0.1)

    def test_clip_eta(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.1,
                                        nb_iter=5)

        delta = np.max(np.abs(x_adv - x_val), axis=1)
        self.assertTrue(np.all(delta <= 1.))

    def test_generate_np_gives_clipped_adversarial_examples(self):
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        x_adv = self.attack.generate_np(x_val, eps=1.0, eps_iter=0.1,
                                        nb_iter=5,
                                        clip_min=-0.2, clip_max=0.3)

        self.assertTrue(-0.201 < np.min(x_adv))
        self.assertTrue(np.max(x_adv) < .301)

    def test_multiple_initial_random_step(self):
        """
        This test generates multiple adversarial examples until an adversarial
        example is generated with a different label compared to the original
        label. This is the procedure suggested in Madry et al. (2017).

        This test will fail if an initial random step is not taken (error>0.5).
        """
        x_val = np.random.rand(100, 2)
        x_val = np.array(x_val, dtype=np.float32)

        orig_labs = np.argmax(self.sess.run(self.model(x_val)), axis=1)
        new_labs_multi = orig_labs.copy()

        # Generate multiple adversarial examples
        for i in range(10):
            x_adv = self.attack.generate_np(x_val, eps=.5, eps_iter=0.05,
                                            clip_min=0.5, clip_max=0.7,
                                            nb_iter=2)
            new_labs = np.argmax(self.sess.run(self.model(x_adv)), axis=1)

            # Examples for which we have not found adversarial examples
            I = (orig_labs == new_labs_multi)
            new_labs_multi[I] = new_labs[I]

        self.assertTrue(np.mean(orig_labs == new_labs_multi) < 0.1)


class TestMultiGPUAdvTrain(CleverHansTest):
    def test_run_multigpu_madryetal(self):
        pass

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

        # Check that the tutorial is deterministic (seeded properly)
        if LooseVersion(tf.__version__) >= LooseVersion('1.1.0'):
            atol_fac = 4
        else:
            atol_fac = 2

        # Run the multi-gpu trainer for clean training
        flags.update({'batch_size': 128, 'learning_rate': 0.001,
                      'dataset': 'mnist', 'only_adv_train': False,
                      'eval_iters': 1, 'ngpu': 1, 'fast_tests': False,
                      'debug_graph': False, 'attack_type_train': None,
                      'save_dir': None, 'save_steps': 10000,
                      'attack_nb_iter_train': None})

        flags.update({'adv_train': False,
                      'save': False,
                      'model_type': 'basic',
                      'attack_type_test': 'FGSM',
                      })
        HParams = namedtuple('HParams', flags.keys())

        hparams = HParams(**flags)
        report_dict = run_trainer(hparams)
        report_2 = AccuracyReport()
        report_2.train_clean_train_clean_eval = report_dict['train']
        report_2.clean_train_clean_eval = report_dict['test']
        report_2.clean_train_adv_eval = report_dict['FGSM']

        # Run the multi-gpu trainer for adversarial training
        flags.update({'adv_train': True,
                      'save': False,
                      'model_type': 'basic',
                      'attack_type_test': 'FGSM',
                      'attack_type_train': 'FGSM',
                      })
        HParams = namedtuple('HParams', flags.keys())

        hparams = HParams(**flags)
        report_dict = run_trainer(hparams)
        report_2.train_adv_train_clean_eval = report_dict['train']
        report_2.adv_train_clean_eval = report_dict['test']
        report_2.adv_train_adv_eval = report_dict['FGSM']

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
                         atol=atol_fac * 2e-1)
        self.assertClose(report.adv_train_adv_eval,
                         report_2.adv_train_adv_eval,
                         atol=atol_fac * 2e-1)


if __name__ == '__main__':
    unittest.main()
