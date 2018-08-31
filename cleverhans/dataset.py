"""Dataset class for CleverHans

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cleverhans.utils_mnist import data_mnist

keras = None  # Only load keras if user tries to use a dataset that requires it


class Dataset(object):
    """Abstract base class representing a dataset.
    """

    # The number of classes in the dataset. Should be specified by subclasses.
    NB_CLASSES = None

    def get_factory(self):
        """Returns a picklable callable that recreates the dataset.
        """

        if hasattr(self, 'args'):
            args = self.args
        else:
            args = []

        if hasattr(self, 'kwargs'):
            kwargs = self.kwargs
        else:
            kwargs = {}

        return Factory(type(self), args, kwargs)


class MNIST(Dataset):
    """The MNIST dataset"""

    NB_CLASSES = 10

    def __init__(self, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False):
        self.kwargs = locals()
        del self.kwargs["self"]
        x_train, y_train, x_test, y_test = data_mnist(train_start=train_start,
                                                      train_end=train_end,
                                                      test_start=test_start,
                                                      test_end=test_end)

        if center:
            x_train = x_train * 2. - 1.
            x_test = x_test * 2. - 1.

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class CIFAR10(Dataset):
    """The CIFAR-10 dataset"""

    NB_CLASSES = 10

    def __init__(self, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False):
        self.kwargs = locals()
        del self.kwargs["self"]
        packed = data_cifar10(train_start=train_start,
                              train_end=train_end,
                              test_start=test_start,
                              test_end=test_end)
        x_train, y_train, x_test, y_test = packed

        if center:
            x_train = x_train * 2. - 1.
            x_test = x_test * 2. - 1.

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class Factory(object):
    """
    A callable that creates an object of the specified type and configuration.
    """

    def __init__(self, cls, args, kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        """Returns the created object.
        """
        return self.cls(*self.args, **self.kwargs)


def data_cifar10(train_start=0, train_end=50000, test_start=0, test_end=10000):
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    global keras
    if keras is None:
        import keras
        from keras.datasets import cifar10
        from keras.utils import np_utils

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
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

    X_train = X_train[train_start:train_end, :, :, :]
    Y_train = Y_train[train_start:train_end, :]
    X_test = X_test[test_start:test_end, :]
    Y_test = Y_test[test_start:test_end, :]

    return X_train, Y_train, X_test, Y_test
