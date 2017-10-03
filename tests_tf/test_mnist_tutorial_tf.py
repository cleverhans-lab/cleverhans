from distutils.version import LooseVersion
import unittest
import numpy as np

from cleverhans.devtools.checks import CleverHansTest


class TestMNISTTutorialTF(CleverHansTest):
    def test_mnist_tutorial_tf(self):

        np.random.seed(42)
        import tensorflow as tf
        tf.set_random_seed(42)

        from cleverhans_tutorials import mnist_tutorial_tf

        # Run the MNIST tutorial on a dataset of reduced size
        test_dataset_indices = {'train_start': 0,
                                'train_end': 5000,
                                'test_start': 0,
                                'test_end': 333,
                                'nb_epochs': 2,
                                'testing': True}
        report = mnist_tutorial_tf.mnist_tutorial(**test_dataset_indices)

        # Check accuracy values contained in the AccuracyReport object
        self.assertGreater(report.train_clean_train_clean_eval, 0.97)
        self.assertLess(report.train_clean_train_adv_eval, 0.036)
        self.assertGreater(report.train_adv_train_clean_eval, 0.93)
        self.assertGreater(report.train_adv_train_adv_eval, 0.4)

        # Check that the tutorial is deterministic (seeded properly)
        if LooseVersion(tf.__version__) >= LooseVersion('1.1.0'):
            atol_fac = 1
        else:
            atol_fac = 2
        report_2 = mnist_tutorial_tf.mnist_tutorial(**test_dataset_indices)
        self.assertClose(report.train_clean_train_clean_eval,
                         report_2.train_clean_train_clean_eval,
                         atol=atol_fac * 5e-3)
        self.assertClose(report.train_clean_train_adv_eval,
                         report_2.train_clean_train_adv_eval,
                         atol=atol_fac * 5e-3)
        self.assertClose(report.train_adv_train_clean_eval,
                         report_2.train_adv_train_clean_eval,
                         atol=atol_fac * 2e-2)
        self.assertClose(report.train_adv_train_adv_eval,
                         report_2.train_adv_train_adv_eval,
                         atol=atol_fac * 2e-1)

if __name__ == '__main__':
    unittest.main()
