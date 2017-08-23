import unittest
import numpy as np


class TestMNISTTutorialJSMA(unittest.TestCase):
    def test_mnist_tutorial_jsma(self):

        np.random.seed(42)
        import tensorflow as tf
        tf.set_random_seed(42)

        from cleverhans_tutorials import mnist_tutorial_jsma

        # Run the MNIST tutorial on a dataset of reduced size
        # and disable visualization.
        jsma_tutorial_args = {'train_start': 0,
                              'train_end': 1000,
                              'test_start': 0,
                              'test_end': 1666,
                              'viz_enabled': False,
                              'source_samples': 1,
                              'nb_epochs': 2}
        report = mnist_tutorial_jsma.mnist_tutorial_jsma(**jsma_tutorial_args)

        # Check accuracy values contained in the AccuracyReport object
        # We already have JSMA tests in test_attacks.py, so just sanity
        # check the values here.
        self.assertTrue(report.clean_train_clean_eval > 0.65)
        self.assertTrue(report.clean_train_adv_eval < 0.25)

        # There is no adversarial training in the JSMA tutorial
        self.assertTrue(report.adv_train_clean_eval == 0.)
        self.assertTrue(report.adv_train_adv_eval == 0.)


if __name__ == '__main__':
    unittest.main()
