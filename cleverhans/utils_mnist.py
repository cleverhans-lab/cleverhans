from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.utils import np_utils
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


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

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
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


def model_mnist(logits=False,input_ph=None, img_rows=28, img_cols=28, nb_filters=64, nb_classes=10):
    """
    Defines MNIST model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also return logits tensor
    :param input_ph: The TensorFlow placeholder for the input (needed if returning logits)
    :return:
    """
    model = Sequential()

    model.add(Dropout(0.2, input_shape=(1, img_rows, img_cols)))
    model.add(Convolution2D(nb_filters, 8, 8,
                            subsample=(2, 2),
                            border_mode="same"
                            ))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters * 2, 6, 6, subsample=(2, 2),
        border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters *2, 5, 5, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nb_classes))
    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model
