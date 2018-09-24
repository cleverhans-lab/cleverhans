"""Serialization functionality.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
import tensorflow as tf

from cleverhans.model import Model
from cleverhans.utils import safe_zip


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

    See cleverhans_tutorials/mnist_tutorial_picklable.py for examples of a
    complete model training, pickling, and unpickling process using
    PicklableVariable.

    See cleverhans.picklable_model for models built using PicklableVariable.
    """

    def __init__(self, *args, **kwargs):
        self.var = tf.Variable(*args, **kwargs)

    def __getstate__(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("PicklableVariable requires a default "
                               "TensorFlow session")
        return {'var': sess.run(self.var)}

    def __setstate__(self, d):
        self.var = tf.Variable(d['var'])
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("PicklableVariable requires a default "
                               "TensorFlow session")
        sess.run(self.var.initializer)


class NoRefModel(Model):
    """
    A Model that can be pickled because it contains no references to any
    Variables (e.g. it identifies Variables only by name).
    The Model must be able to find all of its Variables via get_params
    for them to be pickled.
    Note that NoRefModel may have different Variable names after it is
    restored, e.g. if the unpickling is run with a different enclosing
    scope. NoRefModel will still work in these circumstances as long
    as get_params returns the same order of Variables after unpickling
    as it did before pickling.
    See also cleverhans.picklable_model for a different, complementary
    pickling strategy: models that can be pickled because they use *only*
    references to Variables and work regardless of Variable names.
    """

    def __getstate__(self):
        # Serialize everything except the Variables
        out = self.__dict__.copy()

        # The base Model class adds this tf reference to self
        # We mustn't pickle anything tf, this will need to be
        # regenerated after the model is reloaded.
        if "_dummy_input" in out:
            del out["_dummy_input"]

        # Add the Variables
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("NoRefModel requires a default "
                               "TensorFlow session")
        out["_tf_variables"] = sess.run(self.get_params())
        return out

    def __setstate__(self, d):
        tf_variables = d["_tf_variables"]
        del d["_tf_variables"]
        # Deserialize everything except the Variables
        self.__dict__ = d
        # Deserialize the Variables
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError("NoRefModel requires a default "
                               "TensorFlow session")
        for var, value in safe_zip(self.get_params(), tf_variables):
            var.load(value, sess)


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
