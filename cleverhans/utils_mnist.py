from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import warnings

from . import utils


def data_mnist(datadir='/tmp/'):
    """
    Load and preprocess MNIST dataset
    :return:
    """

    if 'tensorflow' in sys.modules:
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(datadir, one_hot=True, reshape=False)
        X_train = np.vstack((mnist.train.images, mnist.validation.images))
        Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
        X_test = mnist.test.images
        Y_test = mnist.test.labels
    else:
        warnings.warn("cleverhans support for Theano is deprecated and "
                      "will be dropped on 2017-11-08.")
        import keras
        from keras.datasets import mnist
        from keras.utils import np_utils

        # These values are specific to MNIST
        img_rows = 28
        img_cols = 28
        nb_classes = 10

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        if keras.backend.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, Y_train, X_test, Y_test


def model_mnist(logits=False, input_ph=None, img_rows=28, img_cols=28,
                nb_filters=64, nb_classes=10):
    warnings.warn("`utils_mnist.model_mnist` is deprecated. Switch to"
                  "`utils.cnn_model`. `utils_mnist.model_mnist` will "
                  "be removed after 2017-08-17.")
    return utils.cnn_model(logits=logits, input_ph=input_ph,
                           img_rows=img_rows, img_cols=img_cols,
                           nb_filters=nb_filters, nb_classes=nb_classes)
