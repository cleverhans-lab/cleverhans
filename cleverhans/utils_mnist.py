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

import numpy as np

from cleverhans import utils


def maybe_download_mnist_file(file_name, datadir=None, force=False):
    try:
        from urllib.parse import urljoin
        from urllib.request import urlretrieve
    except ImportError:
        from urlparse import urljoin
        from urllib import urlretrieve

    if not datadir:
        datadir = tempfile.gettempdir()
    dest_file = os.path.join(datadir, file_name)

    isfile = os.path.isfile(dest_file)

    if force or not isfile:
        url = urljoin('http://yann.lecun.com/exdb/mnist/', file_name)
        urlretrieve(url, dest_file)
    return dest_file


def download_and_parse_mnist_file(file_name, datadir=None, force=False):
    file_name = maybe_download_mnist_file(file_name, datadir=datadir,
                                          force=force)

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


def data_mnist(datadir='/tmp/', train_start=0, train_end=60000, test_start=0,
               test_end=10000):
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

    Y_train = utils.to_categorical(Y_train, num_classes=10)
    Y_test = utils.to_categorical(Y_test, num_classes=10)
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = data_mnist()
