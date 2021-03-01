"""Utility functions for mocking up tests.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from cleverhans.dataset import Dataset, np_utils


def random_feed_dict(rng, placeholders):
  """
  Returns random data to be used with `feed_dict`.
  :param rng: A numpy.random.RandomState instance
  :param placeholders: List of tensorflow placeholders
  :return: A dict mapping placeholders to random numpy values
  """

  output = {}

  for placeholder in placeholders:
    if placeholder.dtype != 'float32':
      raise NotImplementedError()
    value = rng.randn(*placeholder.shape).astype('float32')
    output[placeholder] = value

  return output

class SimpleDataset(Dataset):
  """
  A dataset containing random values.
  Values are uniformly distributed, either in [0, max_val] or [-1, max_val].
  """

  def __init__(self, dim=2, train_start=0, train_end=3, test_start=0, test_end=5, center=False, max_val=1.,
               nb_classes=5):
    kwargs = copy.copy(locals())
    del kwargs['self']
    if "__class__" in kwargs:
      del kwargs["__class__"]
    super(SimpleDataset, self).__init__(kwargs)
    self.__dict__.update(kwargs)
    train_x_rng = np.random.RandomState([2018, 11, 9, 1])
    # Even if train_start is not 0, we should still generate the first training examples from the rng.
    # This way the dataset looks like it is an array of deterministic data that we index using train_start.
    self.x_train = train_x_rng.uniform(- center * max_val, max_val, (train_end, dim))[train_start:]
    # Use a second rng for the test set so that it also looks like an array of deterministic data that we
    # index into, unaffected by the number of training examples.
    test_x_rng = np.random.RandomState([2018, 11, 9, 2])
    self.x_test = test_x_rng.uniform(- center * max_val, max_val, (test_end, dim))[test_start:]
    # Likewise, to keep the number of examples read from the rng affecting the values of the labels, we
    # must generate the labels from a different rng
    train_y_rng = np.random.RandomState([2018, 11, 9, 3])
    self.y_train = train_y_rng.randint(low=0, high=nb_classes, size=(train_end, 1))[train_start:]
    test_y_rng = np.random.RandomState([2018, 11, 9, 4])
    self.y_test = test_y_rng.randint(low=0, high=nb_classes, size=(test_end, 1))[test_start:]
    assert self.x_train.shape[0] == self.y_train.shape[0]
    assert self.x_test.shape[0] == self.y_test.shape[0]
    self.y_train = np_utils.to_categorical(self.y_train, nb_classes)
    self.y_test = np_utils.to_categorical(self.y_test, nb_classes)
