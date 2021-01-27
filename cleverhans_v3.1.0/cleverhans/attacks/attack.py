"""
The Attack interface.
"""

from abc import ABCMeta
import collections
import warnings

import numpy as np
import tensorflow as tf

from cleverhans.compat import reduce_max
from cleverhans.model import Model
from cleverhans import utils

_logger = utils.create_logger("cleverhans.attacks.attack")


class Attack(object):
  """
  Abstract base class for all attack classes.
  """
  __metaclass__ = ABCMeta

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    :param model: An instance of the cleverhans.model.Model class.
    :param sess: The (possibly optional) tf.Session to run graphs in.
    :param dtypestr: Floating point precision to use (change to float64
                     to avoid numerical instabilities).
    :param back: (deprecated and will be removed on or after 2019-03-26).
                 The backend to use. Currently 'tf' is the only option.
    """
    if 'back' in kwargs:
      if kwargs['back'] == 'tf':
        warnings.warn("Argument back to attack constructors is not needed"
                      " anymore and will be removed on or after 2019-03-26."
                      " All attacks are implemented using TensorFlow.")
      else:
        raise ValueError("Backend argument must be 'tf' and is now deprecated"
                         "It will be removed on or after 2019-03-26.")

    self.tf_dtype = tf.as_dtype(dtypestr)
    self.np_dtype = np.dtype(dtypestr)

    if sess is not None and not isinstance(sess, tf.Session):
      raise TypeError("sess is not an instance of tf.Session")

    from cleverhans import attacks_tf
    attacks_tf.np_dtype = self.np_dtype
    attacks_tf.tf_dtype = self.tf_dtype

    if not isinstance(model, Model):
      raise TypeError("The model argument should be an instance of"
                      " the cleverhans.model.Model class.")

    # Prepare attributes
    self.model = model
    self.sess = sess
    self.dtypestr = dtypestr

    # We are going to keep track of old graphs and cache them.
    self.graphs = {}

    # When calling generate_np, arguments in the following set should be
    # fed into the graph, as they are not structural items that require
    # generating a new graph.
    # This dict should map names of arguments to the types they should
    # have.
    # (Usually, the target class will be a feedable keyword argument.)
    self.feedable_kwargs = tuple()

    # When calling generate_np, arguments in the following set should NOT
    # be fed into the graph, as they ARE structural items that require
    # generating a new graph.
    # This list should contain the names of the structural arguments.
    self.structural_kwargs = []

  def generate(self, x, **kwargs):
    """
    Generate the attack's symbolic graph for adversarial examples. This
    method should be overriden in any child class that implements an
    attack that is expressable symbolically. Otherwise, it will wrap the
    numerical implementation as a symbolic operator.

    :param x: The model's symbolic inputs.
    :param **kwargs: optional parameters used by child classes.
      Each child class defines additional parameters as needed.
      Child classes that use the following concepts should use the following
      names:
        clip_min: minimum feature value
        clip_max: maximum feature value
        eps: size of norm constraint on adversarial perturbation
        ord: order of norm constraint
        nb_iter: number of iterations
        eps_iter: size of norm constraint on iteration
        y_target: if specified, the attack is targeted.
        y: Do not specify if y_target is specified.
           If specified, the attack is untargeted, aims to make the output
           class not be y.
           If neither y_target nor y is specified, y is inferred by having
           the model classify the input.
      For other concepts, it's generally a good idea to read other classes
      and check for name consistency.
    :return: A symbolic representation of the adversarial examples.
    """

    error = "Sub-classes must implement generate."
    raise NotImplementedError(error)
    # Include an unused return so pylint understands the method signature
    return x

  def construct_graph(self, fixed, feedable, x_val, hash_key):
    """
    Construct the graph required to run the attack through generate_np.

    :param fixed: Structural elements that require defining a new graph.
    :param feedable: Arguments that can be fed to the same graph when
                     they take different values.
    :param x_val: symbolic adversarial example
    :param hash_key: the key used to store this graph in our cache
    """
    # try our very best to create a TF placeholder for each of the
    # feedable keyword arguments, and check the types are one of
    # the allowed types
    class_name = str(self.__class__).split(".")[-1][:-2]
    _logger.info("Constructing new graph for attack " + class_name)

    # remove the None arguments, they are just left blank
    for k in list(feedable.keys()):
      if feedable[k] is None:
        del feedable[k]

    # process all of the rest and create placeholders for them
    new_kwargs = dict(x for x in fixed.items())
    for name, value in feedable.items():
      given_type = value.dtype
      if isinstance(value, np.ndarray):
        if value.ndim == 0:
          # This is pretty clearly not a batch of data
          new_kwargs[name] = tf.placeholder(given_type, shape=[], name=name)
        else:
          # Assume that this is a batch of data, make the first axis variable
          # in size
          new_shape = [None] + list(value.shape[1:])
          new_kwargs[name] = tf.placeholder(given_type, new_shape, name=name)
      elif isinstance(value, utils.known_number_types):
        new_kwargs[name] = tf.placeholder(given_type, shape=[], name=name)
      else:
        raise ValueError("Could not identify type of argument " +
                         name + ": " + str(value))

    # x is a special placeholder we always want to have
    x_shape = [None] + list(x_val.shape)[1:]
    x = tf.placeholder(self.tf_dtype, shape=x_shape)

    # now we generate the graph that we want
    x_adv = self.generate(x, **new_kwargs)

    self.graphs[hash_key] = (x, new_kwargs, x_adv)

    if len(self.graphs) >= 10:
      warnings.warn("Calling generate_np() with multiple different "
                    "structural parameters is inefficient and should"
                    " be avoided. Calling generate() is preferred.")

  def generate_np(self, x_val, **kwargs):
    """
    Generate adversarial examples and return them as a NumPy array.
    Sub-classes *should not* implement this method unless they must
    perform special handling of arguments.

    :param x_val: A NumPy array with the original inputs.
    :param **kwargs: optional parameters used by child classes.
    :return: A NumPy array holding the adversarial examples.
    """

    if self.sess is None:
      raise ValueError("Cannot use `generate_np` when no `sess` was"
                       " provided")

    packed = self.construct_variables(kwargs)
    fixed, feedable, _, hash_key = packed

    if hash_key not in self.graphs:
      self.construct_graph(fixed, feedable, x_val, hash_key)
    else:
      # remove the None arguments, they are just left blank
      for k in list(feedable.keys()):
        if feedable[k] is None:
          del feedable[k]

    x, new_kwargs, x_adv = self.graphs[hash_key]

    feed_dict = {x: x_val}

    for name in feedable:
      feed_dict[new_kwargs[name]] = feedable[name]

    return self.sess.run(x_adv, feed_dict)

  def construct_variables(self, kwargs):
    """
    Construct the inputs to the attack graph to be used by generate_np.

    :param kwargs: Keyword arguments to generate_np.
    :return:
      Structural arguments
      Feedable arguments
      Output of `arg_type` describing feedable arguments
      A unique key
    """
    if isinstance(self.feedable_kwargs, dict):
      warnings.warn("Using a dict for `feedable_kwargs is deprecated."
                    "Switch to using a tuple."
                    "It is not longer necessary to specify the types "
                    "of the arguments---we build a different graph "
                    "for each received type."
                    "Using a dict may become an error on or after "
                    "2019-04-18.")
      feedable_names = tuple(sorted(self.feedable_kwargs.keys()))
    else:
      feedable_names = self.feedable_kwargs
      if not isinstance(feedable_names, tuple):
        raise TypeError("Attack.feedable_kwargs should be a tuple, but "
                        "for subclass " + str(type(self)) + " it is "
                        + str(self.feedable_kwargs) + " of type "
                        + str(type(self.feedable_kwargs)))

    # the set of arguments that are structural properties of the attack
    # if these arguments are different, we must construct a new graph
    fixed = dict(
        (k, v) for k, v in kwargs.items() if k in self.structural_kwargs)

    # the set of arguments that are passed as placeholders to the graph
    # on each call, and can change without constructing a new graph
    feedable = {k: v for k, v in kwargs.items() if k in feedable_names}
    for k in feedable:
      if isinstance(feedable[k], (float, int)):
        feedable[k] = np.array(feedable[k])

    for key in kwargs:
      if key not in fixed and key not in feedable:
        raise ValueError(str(type(self)) + ": Undeclared argument: " + key)

    feed_arg_type = arg_type(feedable_names, feedable)

    if not all(isinstance(value, collections.Hashable)
               for value in fixed.values()):
      # we have received a fixed value that isn't hashable
      # this means we can't cache this graph for later use,
      # and it will have to be discarded later
      hash_key = None
    else:
      # create a unique key for this set of fixed paramaters
      hash_key = tuple(sorted(fixed.items())) + tuple([feed_arg_type])

    return fixed, feedable, feed_arg_type, hash_key

  def get_or_guess_labels(self, x, kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' is in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.
    """
    if 'y' in kwargs and 'y_target' in kwargs:
      raise ValueError("Can not set both 'y' and 'y_target'.")
    elif 'y' in kwargs:
      labels = kwargs['y']
    elif 'y_target' in kwargs and kwargs['y_target'] is not None:
      labels = kwargs['y_target']
    else:
      preds = self.model.get_probs(x)
      preds_max = reduce_max(preds, 1, keepdims=True)
      original_predictions = tf.to_float(tf.equal(preds, preds_max))
      labels = tf.stop_gradient(original_predictions)
      del preds
    if isinstance(labels, np.ndarray):
      nb_classes = labels.shape[1]
    else:
      nb_classes = labels.get_shape().as_list()[1]
    return labels, nb_classes

  def parse_params(self, params=None):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    :param params: a dictionary of attack-specific parameters
    :return: True when parsing was successful
    """

    if params is not None:
      warnings.warn("`params` is unused and will be removed "
                    " on or after 2019-04-26.")
    return True


def arg_type(arg_names, kwargs):
  """
  Returns a hashable summary of the types of arg_names within kwargs.
  :param arg_names: tuple containing names of relevant arguments
  :param kwargs: dict mapping string argument names to values.
    These must be values for which we can create a tf placeholder.
    Currently supported: numpy darray or something that can ducktype it
  returns:
    API contract is to return a hashable object describing all
    structural consequences of argument values that can otherwise
    be fed into a graph of fixed structure.
    Currently this is implemented as a tuple of tuples that track:
      - whether each argument was passed
      - whether each argument was passed and not None
      - the dtype of each argument
    Callers shouldn't rely on the exact structure of this object,
    just its hashability and one-to-one mapping between graph structures.
  """
  assert isinstance(arg_names, tuple)
  passed = tuple(name in kwargs for name in arg_names)
  passed_and_not_none = []
  for name in arg_names:
    if name in kwargs:
      passed_and_not_none.append(kwargs[name] is not None)
    else:
      passed_and_not_none.append(False)
  passed_and_not_none = tuple(passed_and_not_none)
  dtypes = []
  for name in arg_names:
    if name not in kwargs:
      dtypes.append(None)
      continue
    value = kwargs[name]
    if value is None:
      dtypes.append(None)
      continue
    assert hasattr(value, 'dtype'), type(value)
    dtype = value.dtype
    if not isinstance(dtype, np.dtype):
      dtype = dtype.as_np_dtype
    assert isinstance(dtype, np.dtype)
    dtypes.append(dtype)
  dtypes = tuple(dtypes)
  return (passed, passed_and_not_none, dtypes)
