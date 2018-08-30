"""Dataset class for CleverHans

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cleverhans.utils_mnist import data_mnist


class Dataset(object):
    """Abstract base class representing a dataset.
    """

    # The number of classes in the dataset. Should be specified by subclasses.
    nb_classes = None

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

    nb_classes = 10

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
