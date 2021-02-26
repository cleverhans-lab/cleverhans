"""Serialization functionality.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import joblib
import tensorflow as tf

from cleverhans.model import Model
from cleverhans.utils import ordered_union
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
  The Model must be able to find all of its Variables via get_vars
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
    tf_variables = self.get_vars()
    out[VARS] = sess.run(tf_variables)
    out[VAR_NAMES] = [var.name for var in tf_variables]
    return out

  def __setstate__(self, d):
    tf_variables = d[VARS]
    del d[VARS]
    tf_variable_names = None
    # older joblib files may not have "_tf_variable_names"
    if VAR_NAMES in d:
      tf_variable_names = d[VAR_NAMES]
      del d[VAR_NAMES]
    else:
      warnings.warn("This joblib file has no " + VAR_NAMES + " field. "
                    "The field may become required on or after 2019-04-11."
                    "You can make your file compatible with the new format by"
                    " loading the file and re-saving it.")
    # Deserialize everything except the Variables
    self.__dict__ = d
    # Deserialize the Variables
    sess = tf.get_default_session()
    if sess is None:
      raise RuntimeError("NoRefModel requires a default "
                         "TensorFlow session")
    cur_vars = self.get_vars()
    if len(cur_vars) != len(tf_variables):
      print("Model format mismatch")
      print("Current model has " + str(len(cur_vars)) + " variables")
      print("Saved model has " + str(len(tf_variables)) + " variables")
      print("Names of current vars:")
      for var in cur_vars:
        print("\t" + var.name)
      if tf_variable_names is not None:
        print("Names of saved vars:")
        for name in tf_variable_names:
          print("\t" + name)
      else:
        print("Saved vars use old format, no names available for them")
      assert False

    found = [False] * len(cur_vars)
    if tf_variable_names is not None:
      # New version using the names to handle changes in ordering
      for value, name in safe_zip(tf_variables, tf_variable_names):
        value_found = False
        for idx, cur_var in enumerate(cur_vars):
          if cur_var.name == name:
            assert not found[idx]
            value_found = True
            found[idx] = True
            cur_var.load(value, sess)
            break
        assert value_found
      assert all(found)
    else:
      # Old version that works if and only if the order doesn't change
      for var, value in safe_zip(cur_vars, tf_variables):
        var.load(value, sess)

  def get_vars(self):
    """
    Provides access to the model's Variables.
    This may include Variables that are not parameters, such as batch
    norm running moments.
    :return: A list of all Variables defining the model.
    """

    # Catch eager execution and assert function overload.
    try:
      if tf.executing_eagerly():
        raise NotImplementedError("For Eager execution - get_vars "
                                  "must be overridden.")
    except AttributeError:
      pass

    done = False
    tried_to_make_params = False
    while not done:
      # Most models in cleverhans use only trainable variables and do not
      # make sure the other collections are updated correctly.
      trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         self.scope + "/")
      # When wrapping other code, such as the CIFAR 10 challenge models,
      # we need to make sure we get the batch norm running averages as well
      # as the trainable variables.
      model_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES,
                                     self.scope + "/")
      scope_vars = ordered_union(trainable_vars, model_vars)

      if len(scope_vars) > 0:
        done = True
      else:
        assert not tried_to_make_params
        tried_to_make_params = True
        self.make_params()

    # Make sure no variables have been added or removed
    if hasattr(self, "num_vars"):
      assert self.num_vars == len(scope_vars)
    else:
      self.num_vars = len(scope_vars)

    return scope_vars


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

VARS = "_tf_variables"
VAR_NAMES = "_tf_variable_names"
