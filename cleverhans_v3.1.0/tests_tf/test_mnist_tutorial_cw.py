# pylint: disable=missing-docstring
import unittest
import numpy as np
from cleverhans.devtools.checks import CleverHansTest


class TestMNISTTutorialCW(CleverHansTest):
  def test_mnist_tutorial_cw(self):
    import tensorflow as tf
    from cleverhans_tutorials import mnist_tutorial_cw

    # Run the MNIST tutorial on a dataset of reduced size
    # and disable visualization.
    cw_tutorial_args = {'train_start': 0,
                        'train_end': 10000,
                        'test_start': 0,
                        'test_end': 1666,
                        'viz_enabled': False}
    g = tf.Graph()
    with g.as_default():
      np.random.seed(42)
      report = mnist_tutorial_cw.mnist_tutorial_cw(**cw_tutorial_args)

    # Check accuracy values contained in the AccuracyReport object
    self.assertGreater(report.clean_train_clean_eval, 0.85)
    self.assertEqual(report.clean_train_adv_eval, 0.00)

    # There is no adversarial training in the CW tutorial
    self.assertEqual(report.adv_train_clean_eval, 0.)
    self.assertEqual(report.adv_train_adv_eval, 0.)

    g = tf.Graph()
    with g.as_default():
      np.random.seed(42)
      report_2 = mnist_tutorial_cw.mnist_tutorial_cw(**cw_tutorial_args)

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
