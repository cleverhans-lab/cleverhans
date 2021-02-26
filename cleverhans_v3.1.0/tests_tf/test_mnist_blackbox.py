# pylint: disable=missing-docstring
import unittest
import numpy as np
# pylint bug on next line
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module
from cleverhans.devtools.checks import CleverHansTest

HAS_GPU = 'GPU' in {x.device_type for x in device_lib.list_local_devices()}


class TestMNISTBlackboxF(CleverHansTest):
  def test_mnist_blackbox(self):
    import tensorflow as tf
    from cleverhans_tutorials import mnist_blackbox

    # Run the MNIST tutorial on a dataset of reduced size, reduced number
    # of data augmentations, increased substitute holdout for faster runtime.
    mnist_blackbox_args = {'train_start': 0,
                           'train_end': 5000,
                           'test_start': 0,
                           'test_end': 2000,
                           'data_aug': 1,
                           'holdout': 1000,
                           'nb_epochs': 2,
                           'nb_epochs_s': 6}
    g = tf.Graph()
    with g.as_default():
      np.random.seed(42)
      report = mnist_blackbox.mnist_blackbox(**mnist_blackbox_args)

    # Check accuracy values contained in the AccuracyReport object
    self.assertTrue(report['bbox'] > 0.7, report['bbox'])
    self.assertTrue(report['sub'] > 0.7, report['sub'])
    self.assertTrue(report['bbox_on_sub_adv_ex'] < 0.3,
                    report['bbox_on_sub_adv_ex'])

    g = tf.Graph()
    with g.as_default():
      np.random.seed(42)
      report_2 = mnist_blackbox.mnist_blackbox(**mnist_blackbox_args)

    atol_fac = 1e-2 if HAS_GPU else 1e-6
    self.assertClose(report['bbox'], report_2['bbox'], atol=atol_fac * 1)
    self.assertClose(report['sub'], report_2['sub'], atol=atol_fac * 1)
    self.assertClose(report['bbox_on_sub_adv_ex'],
                     report_2['bbox_on_sub_adv_ex'], atol=atol_fac * 1)


if __name__ == '__main__':
  unittest.main()
