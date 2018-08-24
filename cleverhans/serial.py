"""Serialization functionality.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import joblib


class PicklableVariable(object):
    """
    A wrapper around a Variable that makes it picklable.

    The name of the Variable will not be reliable, only the value. Models
    intended to be picklable should identify variables by referencing
    Python objects rather than by using TensorFlow's names.

    TensorFlow Variables have different values associated with each Session.
    For this class, the value associated with the default Session will be used
    for both saving and loading, so both operations require that a default
    Session has been selected.

    Pickle is not secure. Unpickle only files you made yourself.
    """

    def __init__(self, *args, **kwargs):
        self.var = tf.Variable(*args, **kwargs)

    def __getstate__(self):
        sess = tf.get_default_session()
        return {'var': sess.run(self.var)}

    def __setstate__(self, d):
        self.var = tf.Variable(d['var'])
        sess = tf.get_default_session()
        sess.run(self.var.initializer)


def save(filepath, obj):
    """Saves an object to the specified filepath using joblib.

    joblib is like pickle but will save NumPy arrays as separate files for
    greater efficiency.

    :param filepath: str, path to save to
    :obj filepath: object to save
    """

    joblib.dump(obj, filepath)


def load(filepath):
    """Returns an object stored via `save`
    """

    obj = joblib.load(filepath)

    return obj
