# pylint: disable=missing-docstring
import unittest
import numpy as np
# pylint bug on next line
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module
from cleverhans.devtools.checks import CleverHansTest

HAS_GPU = 'GPU' in {x.device_type for x in device_lib.list_local_devices()}


class TestMNISTTutorialKeras(CleverHansTest):
  def test_mnist_tutorial_keras(self):

    import tensorflow as tf
    from cleverhans_tutorials import mnist_tutorial_keras

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
      report = mnist_tutorial_keras.mnist_tutorial(**test_dataset_indices)

    # Check accuracy values contained in the AccuracyReport object
    self.assertTrue(report.train_clean_train_clean_eval > 0.90)
    self.assertTrue(report.train_clean_train_adv_eval < 0.05)
    self.assertTrue(report.train_adv_train_clean_eval > 0.90)
    self.assertTrue(report.train_adv_train_adv_eval > 0.30)

    atol_fac = 5e-2 if HAS_GPU else 1e-6
    g = tf.Graph()
    with g.as_default():
      np.random.seed(42)
      report_2 = mnist_tutorial_keras.mnist_tutorial(**test_dataset_indices)

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
