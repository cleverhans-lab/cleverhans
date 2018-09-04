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


class MNIST(Dataset):
    """The MNIST dataset"""

    NB_CLASSES = 10

    def __init__(self, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False):
        super(MNIST, self).__init__(locals())
        x_train, y_train, x_test, y_test = data_mnist(train_start=train_start,
                                                      train_end=train_end,
                                                      test_start=test_start,
                                                      test_end=test_end)

        if center:
            x_train = x_train * 2. - 1.
            x_test = x_test * 2. - 1.

        self.x_train = x_train.astype('float32')
        self.y_train = y_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.y_test = y_test.astype('float32')


class CIFAR10(Dataset):
    """The CIFAR-10 dataset"""

    NB_CLASSES = 10

    def __init__(self, train_start=0, train_end=60000, test_start=0,
                 test_end=10000, center=False):
        super(CIFAR10, self).__init__(locals())
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

    def __init__(self, cls, kwargs):
        self.cls = cls
        self.kwargs = kwargs

    def __call__(self):
        """Returns the created object.
        """
        return self.cls(**self.kwargs)


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
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
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
