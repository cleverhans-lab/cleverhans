from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras.datasets import mnist
from keras.utils import np_utils
import warnings

from . import utils


def data_mnist():
    """
    Preprocess MNIST dataset
    :return:
    """

    # These values are specific to MNIST
    img_rows = 28
    img_cols = 28
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def model_mnist(logits=False, input_ph=None, img_rows=28, img_cols=28,
                nb_filters=64, nb_classes=10):
    warnings.warn("`utils_mnist.model_mnist` is deprecated. Switch to"
                  "`utils.cnn_model`. `utils_mnist.model_mnist` will "
                  "be removed after 2017-08-17.")
    return utils.cnn_model(logits=logits, input_ph=input_ph,
                           img_rows=img_rows, img_cols=img_cols,
                           nb_filters=nb_filters, nb_classes=nb_classes)
