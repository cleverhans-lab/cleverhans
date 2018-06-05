import unittest

import numpy as np

from cleverhans.devtools.checks import CleverHansTest


class TestMNISTTutorialPytorch(CleverHansTest):
    def test_mnist_tutorial_pytorch(self):
        import tensorflow as tf
        from cleverhans_tutorials import mnist_tutorial_pytorch

        # Run the MNIST tutorial on a dataset of reduced size
        with tf.Graph().as_default():
            np.random.seed(42)
            report = mnist_tutorial_pytorch.mnist_tutorial(nb_epochs=1)

        # Check accuracy values contained in the AccuracyReport object
        self.assertGreater(report.clean_train_clean_eval, 0.97)
        self.assertLess(report.clean_train_adv_eval, 0.10)


if __name__ == '__main__':
    unittest.main()
