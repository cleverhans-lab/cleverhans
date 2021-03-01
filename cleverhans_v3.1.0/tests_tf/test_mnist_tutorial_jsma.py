# pylint: disable=missing-docstring
import unittest
import numpy as np
from cleverhans.devtools.checks import CleverHansTest


class TestMNISTTutorialJSMA(CleverHansTest):
  def test_mnist_tutorial_jsma(self):

    import tensorflow as tf
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
    g = tf.Graph()
    with g.as_default():
      np.random.seed(42)
      report = mnist_tutorial_jsma.mnist_tutorial_jsma(**jsma_tutorial_args)

    # Check accuracy values contained in the AccuracyReport object
    # We already have JSMA tests in test_attacks.py, so just sanity
    # check the values here.
    self.assertTrue(report.clean_train_clean_eval > 0.65)
    self.assertTrue(report.clean_train_adv_eval < 0.25)

    # There is no adversarial training in the JSMA tutorial
    self.assertTrue(report.adv_train_clean_eval == 0.)
    self.assertTrue(report.adv_train_adv_eval == 0.)

    g = tf.Graph()
    with g.as_default():
      np.random.seed(42)
      report_2 = mnist_tutorial_jsma.mnist_tutorial_jsma(**jsma_tutorial_args)

    atol_fac = 1e-6
    self.assertClose(report.train_clean_train_clean_eval,
                     report_2.train_clean_train_clean_eval,
                     atol=atol_fac * 1)
    self.assertClose(report.train_clean_train_adv_eval,
                     report_2.train_clean_train_adv_eval,
                     atol=atol_fac * 1)
    self.assertClose(report.train_adv_train_clean_eval,
                     report_2.train_adv_train_clean_eval,
                     atol=atol_fac * 1)
    self.assertClose(report.train_adv_train_adv_eval,
                     report_2.train_adv_train_adv_eval,
                     atol=atol_fac * 1)


if __name__ == '__main__':
  unittest.main()
