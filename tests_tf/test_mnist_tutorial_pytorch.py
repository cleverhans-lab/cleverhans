import unittest

import numpy as np
from tensorflow.python.client import device_lib

from cleverhans.devtools.checks import CleverHansTest

HAS_GPU = 'GPU' in set([x.device_type for x in device_lib.list_local_devices()])


class TestMNISTTutorialTF(CleverHansTest):
    def test_mnist_tutorial_pytorch(self):
        import tensorflow as tf
        from cleverhans_tutorials import mnist_tutorial_pytorch

        # Run the MNIST tutorial on a dataset of reduced size
        test_dataset_indices = {
            'nb_epochs': 1,
            'testing': True}
        g = tf.Graph()

        with g.as_default():
            np.random.seed(42)
            report = mnist_tutorial_pytorch.mnist_tutorial(
                **test_dataset_indices)

        # Check accuracy values contained in the AccuracyReport object
        self.assertGreater(report.clean_train_clean_eval, 0.97)
        self.assertLess(report.clean_train_adv_eval, 0.10)


if __name__ == '__main__':
    unittest.main()
