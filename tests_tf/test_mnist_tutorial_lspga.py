import unittest
import numpy as np
from tensorflow.python.client import device_lib
from cleverhans.devtools.checks import CleverHansTest

HAS_GPU = 'GPU' in set([x.device_type for
                        x in device_lib.list_local_devices()])


class TestMNISTTutorialLSPGA(CleverHansTest):
    def test_mnist_tutorial_lspga(self):

        import tensorflow as tf
        from cleverhans_tutorials import mnist_tutorial_lspga

        # Run the MNIST tutorial on a dataset of reduced size
        test_dataset_indices = {'train_start': 0,
                                'train_end': 5000,
                                'test_start': 0,
                                'test_end': 333,
                                'nb_epochs': 2,
                                'testing': True}
        g = tf.Graph()
        with g.as_default():
            np.random.seed(42)
            report = mnist_tutorial_lspga.mnist_tutorial(
                levels=4, steps=1,
                **test_dataset_indices)

        # Check accuracy values contained in the AccuracyReport object
        self.assertGreater(report.train_clean_train_clean_eval, 0.96)
        self.assertLess(report.train_clean_train_adv_eval, 0.038)
        self.assertGreater(report.train_adv_train_clean_eval, 0.93)
        self.assertGreater(report.train_adv_train_adv_eval, 0.4)


if __name__ == '__main__':
    unittest.main()
