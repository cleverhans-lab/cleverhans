"""
Generic utility functions useful for writing Python code in general
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import warnings
import logging
import os
import re
import subprocess

import numpy as np
from six.moves import xrange

known_number_types = (int, float, np.float16, np.float32, np.float64,
                      np.int8, np.int16, np.int32, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64)


CLEVERHANS_ROOT = os.path.dirname(os.path.dirname(__file__))


class _ArgsWrapper(object):

  """
  Wrapper that allows attribute access to dictionaries
  """

  def __init__(self, args):
    if not isinstance(args, dict):
      args = vars(args)
    self.args = args

  def __getattr__(self, name):
    return self.args.get(name)


class AccuracyReport(object):

  """
  An object summarizing the accuracy results for experiments involving
  training on clean examples or adversarial examples, then evaluating
  on clean or adversarial examples.
  """

  def __init__(self):
    self.clean_train_clean_eval = 0.
    self.clean_train_adv_eval = 0.
    self.adv_train_clean_eval = 0.
    self.adv_train_adv_eval = 0.

    # Training data accuracy results to be used by tutorials
    self.train_clean_train_clean_eval = 0.
    self.train_clean_train_adv_eval = 0.
    self.train_adv_train_clean_eval = 0.
    self.train_adv_train_adv_eval = 0.


def batch_indices(batch_nb, data_length, batch_size):
  """
  This helper function computes a batch start and end index
  :param batch_nb: the batch number
  :param data_length: the total length of the data being parsed by batches
  :param batch_size: the number of inputs in each batch
  :return: pair of (start, end) indices
  """
  # Batch start and end index
  start = int(batch_nb * batch_size)
  end = int((batch_nb + 1) * batch_size)

  # When there are not enough inputs left, we reuse some to complete the
  # batch
  if end > data_length:
    shift = end - data_length
    start -= shift
    end -= shift

  return start, end


def other_classes(nb_classes, class_ind):
  """
  Returns a list of class indices excluding the class indexed by class_ind
  :param nb_classes: number of classes in the task
  :param class_ind: the class index to be omitted
  :return: list of class indices excluding the class indexed by class_ind
  """
  if class_ind < 0 or class_ind >= nb_classes:
    error_str = "class_ind must be within the range (0, nb_classes - 1)"
    raise ValueError(error_str)

  other_classes_list = list(range(nb_classes))
  other_classes_list.remove(class_ind)

  return other_classes_list


def to_categorical(y, nb_classes, num_classes=None):
  """
  Converts a class vector (integers) to binary class matrix.
  This is adapted from the Keras function with the same name.
  :param y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
  :param nb_classes: nb_classes: total number of classes.
  :param num_classses: depricated version of nb_classes
  :return: A binary matrix representation of the input.
  """
  if num_classes is not None:
    if nb_classes is not None:
      raise ValueError("Should not specify both nb_classes and its deprecated "
                       "alias, num_classes")
    warnings.warn("`num_classes` is deprecated. Switch to `nb_classes`."
                  " `num_classes` may be removed on or after 2019-04-23.")
    nb_classes = num_classes
    del num_classes
  y = np.array(y, dtype='int').ravel()
  n = y.shape[0]
  categorical = np.zeros((n, nb_classes))
  categorical[np.arange(n), y] = 1
  return categorical


def random_targets(gt, nb_classes):
  """
  Take in an array of correct labels and randomly select a different label
  for each label in the array. This is typically used to randomly select a
  target class in targeted adversarial examples attacks (i.e., when the
  search algorithm takes in both a source class and target class to compute
  the adversarial example).
  :param gt: the ground truth (correct) labels. They can be provided as a
             1D vector or 2D array of one-hot encoded labels.
  :param nb_classes: The number of classes for this task. The random class
                     will be chosen between 0 and nb_classes such that it
                     is different from the correct class.
  :return: A numpy array holding the randomly-selected target classes
           encoded as one-hot labels.
  """
  # If the ground truth labels are encoded as one-hot, convert to labels.
  if len(gt.shape) == 2:
    gt = np.argmax(gt, axis=1)

  # This vector will hold the randomly selected labels.
  result = np.zeros(gt.shape, dtype=np.int32)

  for class_ind in xrange(nb_classes):
    # Compute all indices in that class.
    in_cl = gt == class_ind
    size = np.sum(in_cl)

    # Compute the set of potential targets for this class.
    potential_targets = other_classes(nb_classes, class_ind)

    # Draw with replacement random targets among the potential targets.
    result[in_cl] = np.random.choice(potential_targets, size=size)

  # Encode vector of random labels as one-hot labels.
  result = to_categorical(result, nb_classes)
  result = result.astype(np.int32)

  return result


def pair_visual(*args, **kwargs):
  """Deprecation wrapper"""
  warnings.warn("`pair_visual` has moved to `cleverhans.plot.pyplot_image`. "
                "cleverhans.utils.pair_visual may be removed on or after "
                "2019-04-24.")
  from cleverhans.plot.pyplot_image import pair_visual as new_pair_visual
  return new_pair_visual(*args, **kwargs)


def grid_visual(*args, **kwargs):
  """Deprecation wrapper"""
  warnings.warn("`grid_visual` has moved to `cleverhans.plot.pyplot_image`. "
                "cleverhans.utils.grid_visual may be removed on or after "
                "2019-04-24.")
  from cleverhans.plot.pyplot_image import grid_visual as new_grid_visual
  return new_grid_visual(*args, **kwargs)


def get_logits_over_interval(*args, **kwargs):
  """Deprecation wrapper"""
  warnings.warn("`get_logits_over_interval` has moved to "
                "`cleverhans.plot.pyplot_image`. "
                "cleverhans.utils.get_logits_over_interval may be removed on "
                "or after 2019-04-24.")
  # pylint:disable=line-too-long
  from cleverhans.plot.pyplot_image import get_logits_over_interval as new_get_logits_over_interval
  return new_get_logits_over_interval(*args, **kwargs)


def linear_extrapolation_plot(*args, **kwargs):
  """Deprecation wrapper"""
  warnings.warn("`linear_extrapolation_plot` has moved to "
                "`cleverhans.plot.pyplot_image`. "
                "cleverhans.utils.linear_extrapolation_plot may be removed on "
                "or after 2019-04-24.")
  # pylint:disable=line-too-long
  from cleverhans.plot.pyplot_image import linear_extrapolation_plot as new_linear_extrapolation_plot
  return new_linear_extrapolation_plot(*args, **kwargs)


def set_log_level(level, name="cleverhans"):
  """
  Sets the threshold for the cleverhans logger to level
  :param level: the logger threshold. You can find values here:
                https://docs.python.org/2/library/logging.html#levels
  :param name: the name used for the cleverhans logger
  """
  logging.getLogger(name).setLevel(level)


def get_log_level(name="cleverhans"):
  """
  Gets the current threshold for the cleverhans logger
  :param name: the name used for the cleverhans logger
  """
  return logging.getLogger(name).getEffectiveLevel()


class TemporaryLogLevel(object):
  """
  A ContextManager that changes a log level temporarily.

  Note that the log level will be set back to its original value when
  the context manager exits, even if the log level has been changed
  again in the meantime.
  """

  def __init__(self, level, name):
    self.name = name
    self.level = level

  def __enter__(self):
    self.old_level = get_log_level(self.name)
    set_log_level(self.level, self.name)

  def __exit__(self, type, value, traceback):
    set_log_level(self.old_level, self.name)
    return True


def create_logger(name):
  """
  Create a logger object with the given name.

  If this is the first time that we call this method, then initialize the
  formatter.
  """
  base = logging.getLogger("cleverhans")
  if len(base.handlers) == 0:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] ' +
                                  '%(message)s')
    ch.setFormatter(formatter)
    base.addHandler(ch)

  return base


def deterministic_dict(normal_dict):
  """
  Returns a version of `normal_dict` whose iteration order is always the same
  """
  out = OrderedDict()
  for key in sorted(normal_dict.keys()):
    out[key] = normal_dict[key]
  return out


def ordered_union(l1, l2):
  """
  Return the union of l1 and l2, with a deterministic ordering.
  (Union of python sets does not necessarily have a consisten iteration
  order)
  :param l1: list of items
  :param l2: list of items
  :returns: list containing one copy of each item that is in l1 or in l2
  """
  out = []
  for e in l1 + l2:
    if e not in out:
      out.append(e)
  return out


def safe_zip(*args):
  """like zip but with these properties:
  - returns a list, rather than an iterator. This is the old Python2 zip behavior.
  - a guarantee that all arguments are the same length.
  (normal zip silently drops entries to make them the same length)
  """
  length = len(args[0])
  if not all(len(arg) == length for arg in args):
    raise ValueError("Lengths of arguments do not match: "
                     + str([len(arg) for arg in args]))
  return list(zip(*args))


def shell_call(command, **kwargs):
  """Calls shell command with argument substitution.

  Args:
    command: command represented as a list. Each element of the list is one
      token of the command. For example "cp a b" becomes ['cp', 'a', 'b']
      If any element of the list looks like '${NAME}' then it will be replaced
      by value from **kwargs with key 'NAME'.
    **kwargs: dictionary with argument substitution

  Returns:
    output of the command

  Raises:
    subprocess.CalledProcessError if command return value is not zero

  This function is useful when you need to do variable substitution prior
  running the command. Below are few examples of how it works:

    shell_call(['cp', 'a', 'b'], a='asd') calls command 'cp a b'

    shell_call(['cp', '${a}', 'b'], a='asd') calls command 'cp asd b',
    '${a}; was replaced with 'asd' before calling the command
  """
  # Regular expression to find instances of '${NAME}' in a string
  CMD_VARIABLE_RE = re.compile('^\\$\\{(\\w+)\\}$')
  command = list(command)
  for i in range(len(command)):
    m = CMD_VARIABLE_RE.match(command[i])
    if m:
      var_id = m.group(1)
      if var_id in kwargs:
        command[i] = kwargs[var_id]
  str_command = ' '.join(command)
  logging.debug('Executing shell command: %s' % str_command)
  return subprocess.check_output(command)

def deep_copy(numpy_dict):
  """
  Returns a copy of a dictionary whose values are numpy arrays.
  Copies their values rather than copying references to them.
  """
  out = {}
  for key in numpy_dict:
    out[key] = numpy_dict[key].copy()
  return out
