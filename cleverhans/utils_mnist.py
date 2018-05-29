from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from . import utils


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

    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home=datadir)
    X_train = mnist.data[:60000] / 255.
    Y_train = mnist.target[:60000]
    X_test = mnist.data[60000:] / 255.
    Y_test = mnist.target[60000:]

    X_train = np.reshape(X_train, (60000, 28, 28, 1))
    X_test = np.reshape(X_test, (10000, 28, 28, 1))

    print(X_train.max())
    print(X_train.min())

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    Y_train = utils.to_categorical(Y_train, num_classes=10)
    Y_test = utils.to_categorical(Y_test, num_classes=10)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test
