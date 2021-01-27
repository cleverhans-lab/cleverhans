# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
import warnings

from cleverhans import dataset

utils_mnist_warning = "cleverhans.utils_mnist is deprecrated and will be " \
                      "removed on or after 2019-03-26. Switch to " \
                      "cleverhans.dataset instead."


def maybe_download_mnist_file(file_name, datadir=None, force=False):
  warnings.warn(utils_mnist_warning)
  url = os.path.join('http://yann.lecun.com/exdb/mnist/', file_name)
  return dataset.maybe_download_file(url, datadir=None, force=False)


def download_and_parse_mnist_file(file_name, datadir=None, force=False):
  warnings.warn(utils_mnist_warning)
  return dataset.download_and_parse_mnist_file(file_name, datadir=None,
                                               force=False)


def data_mnist(datadir=tempfile.gettempdir(), train_start=0,
               train_end=60000, test_start=0, test_end=10000):
  warnings.warn(utils_mnist_warning)
  mnist = dataset.MNIST(train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        center=False)
  return mnist.get_set('train') + mnist.get_set('test')
