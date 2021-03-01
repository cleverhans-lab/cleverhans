"""Dataset class for CleverHans

"""
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import array
import functools
import gzip
import operator
import os
import struct
import tempfile
import sys
import warnings

import numpy as np
import tensorflow as tf
try:
  from tensorflow.python.keras.utils import np_utils
  from tensorflow.keras.datasets import cifar10
except ImportError:
  # In tf 1.8, np_utils doesn't seem to be publicly exposed.
  # In later tf versions, it is, and in pre-tf keras it was too.
  from tensorflow.python.keras import _impl
  np_utils = _impl.keras.utils.np_utils
  # In tf 1.8, "from tensorflow.keras.datasets import cifar10" doesn't work even though the module exists
  cifar10 = tf.keras.datasets.cifar10
  warnings.warn("Support for TensorFlow versions prior to 1.12 is deprecated."
                " CleverHans using earlier versions may quit working on or after 2019-07-07.")
from cleverhans import utils



class Dataset(object):
  """Abstract base class representing a dataset.
  """

  # The number of classes in the dataset. Should be specified by subclasses.
  NB_CLASSES = None

  def __init__(self, kwargs=None):
    if kwargs is None:
      kwargs = {}
    if "self" in kwargs:
      del kwargs["self"]
    self.kwargs = kwargs

  def get_factory(self):
    """Returns a picklable callable that recreates the dataset.
    """

    return Factory(type(self), self.kwargs)

  def get_set(self, which_set):
    """Returns the training set or test set as an (x_data, y_data) tuple.
    :param which_set: 'train' or 'test'
    """
    return (getattr(self, 'x_' + which_set),
            getattr(self, 'y_' + which_set))

  def to_tensorflow(self):
    raise NotImplementedError()

  @classmethod
  def in_memory_dataset(cls, x, y, shuffle=None, repeat=True):
    assert x.shape[0] == y.shape[0]
    d = tf.data.Dataset.range(x.shape[0])
    if repeat:
      d = d.repeat()
    if shuffle:
      d = d.shuffle(shuffle)

    def lookup(p):
      return x[p], y[p]
    d = d.map(lambda i: tf.py_func(lookup, [i], [tf.float32] * 2))
    return d


class MNIST(Dataset):
  """The MNIST dataset"""

  NB_CLASSES = 10

  def __init__(self, train_start=0, train_end=60000, test_start=0,
               test_end=10000, center=False, max_val=1.):
    kwargs = locals()
    if '__class__' in kwargs:
      del kwargs['__class__']
    super(MNIST, self).__init__(kwargs)
    x_train, y_train, x_test, y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    if center:
      x_train = x_train * 2. - 1.
      x_test = x_test * 2. - 1.
    x_train *= max_val
    x_test *= max_val

    self.x_train = x_train.astype('float32')
    self.y_train = y_train.astype('float32')
    self.x_test = x_test.astype('float32')
    self.y_test = y_test.astype('float32')

  def to_tensorflow(self, shuffle=4096):
    return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
            self.in_memory_dataset(self.x_test, self.y_test, repeat=False))


class CIFAR10(Dataset):
  """The CIFAR-10 dataset"""

  NB_CLASSES = 10

  LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                 "frog", "horse", "ship", "truck"]

  def __init__(self, train_start=0, train_end=60000, test_start=0,
               test_end=10000, center=False, max_val=1.):
    kwargs = locals()
    if '__class__' in kwargs:
      del kwargs['__class__']
    super(CIFAR10, self).__init__(kwargs)
    packed = data_cifar10(train_start=train_start,
                          train_end=train_end,
                          test_start=test_start,
                          test_end=test_end)
    x_train, y_train, x_test, y_test = packed

    if center:
      x_train = x_train * 2. - 1.
      x_test = x_test * 2. - 1.
    x_train *= max_val
    x_test *= max_val

    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.max_val = max_val

  def to_tensorflow(self, shuffle=4096):
    # This is much more efficient with data augmentation, see tutorials.
    return (self.in_memory_dataset(self.x_train, self.y_train, shuffle),
            self.in_memory_dataset(self.x_test, self.y_test, repeat=False))


class Factory(object):
  """
  A callable that creates an object of the specified type and configuration.
  """

  def __init__(self, cls, kwargs):
    self.cls = cls
    self.kwargs = kwargs

  def __call__(self):
    """Returns the created object.
    """
    return self.cls(**self.kwargs)


def maybe_download_file(url, datadir=None, force=False):
  try:
    from urllib.request import urlretrieve
  except ImportError:
    from urllib import urlretrieve

  if not datadir:
    datadir = tempfile.gettempdir()
  file_name = url[url.rfind("/")+1:]
  dest_file = os.path.join(datadir, file_name)

  isfile = os.path.isfile(dest_file)

  if force or not isfile:
    urlretrieve(url, dest_file)
  return dest_file


def download_and_parse_mnist_file(file_name, datadir=None, force=False):
  url = os.path.join('http://yann.lecun.com/exdb/mnist/', file_name)
  file_name = maybe_download_file(url, datadir=datadir, force=force)

  # Open the file and unzip it if necessary
  if os.path.splitext(file_name)[1] == '.gz':
    open_fn = gzip.open
  else:
    open_fn = open

  # Parse the file
  with open_fn(file_name, 'rb') as file_descriptor:
    header = file_descriptor.read(4)
    assert len(header) == 4

    zeros, data_type, n_dims = struct.unpack('>HBB', header)
    assert zeros == 0

    hex_to_data_type = {
        0x08: 'B',
        0x09: 'b',
        0x0b: 'h',
        0x0c: 'i',
        0x0d: 'f',
        0x0e: 'd'}
    data_type = hex_to_data_type[data_type]

    # data_type unicode to ascii conversion (Python2 fix)
    if sys.version_info[0] < 3:
      data_type = data_type.encode('ascii', 'ignore')

    dim_sizes = struct.unpack(
        '>' + 'I' * n_dims,
        file_descriptor.read(4 * n_dims))

    data = array.array(data_type, file_descriptor.read())
    data.byteswap()

    desired_items = functools.reduce(operator.mul, dim_sizes)
    assert len(data) == desired_items
    return np.array(data).reshape(dim_sizes)


def data_mnist(datadir=tempfile.gettempdir(), train_start=0,
               train_end=60000, test_start=0, test_end=10000):
  """
  Load and preprocess MNIST dataset
  :param datadir: path to folder where data should be stored
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :return: tuple of four arrays containing training data, training labels,
           testing data and testing labels.
  """
  assert isinstance(train_start, int)
  assert isinstance(train_end, int)
  assert isinstance(test_start, int)
  assert isinstance(test_end, int)

  X_train = download_and_parse_mnist_file(
      'train-images-idx3-ubyte.gz', datadir=datadir) / 255.
  Y_train = download_and_parse_mnist_file(
      'train-labels-idx1-ubyte.gz', datadir=datadir)
  X_test = download_and_parse_mnist_file(
      't10k-images-idx3-ubyte.gz', datadir=datadir) / 255.
  Y_test = download_and_parse_mnist_file(
      't10k-labels-idx1-ubyte.gz', datadir=datadir)

  X_train = np.expand_dims(X_train, -1)
  X_test = np.expand_dims(X_test, -1)

  X_train = X_train[train_start:train_end]
  Y_train = Y_train[train_start:train_end]
  X_test = X_test[test_start:test_end]
  Y_test = Y_test[test_start:test_end]

  Y_train = utils.to_categorical(Y_train, nb_classes=10)
  Y_test = utils.to_categorical(Y_test, nb_classes=10)
  return X_train, Y_train, X_test, Y_test


def data_cifar10(train_start=0, train_end=50000, test_start=0, test_end=10000):
  """
  Preprocess CIFAR10 dataset
  :return:
  """


  # These values are specific to CIFAR10
  img_rows = 32
  img_cols = 32
  nb_classes = 10

  # the data, shuffled and split between train and test sets
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = np_utils.to_categorical(y_train, nb_classes)
  y_test = np_utils.to_categorical(y_test, nb_classes)

  x_train = x_train[train_start:train_end, :, :, :]
  y_train = y_train[train_start:train_end, :]
  x_test = x_test[test_start:test_end, :]
  y_test = y_test[test_start:test_end, :]

  return x_train, y_train, x_test, y_test
