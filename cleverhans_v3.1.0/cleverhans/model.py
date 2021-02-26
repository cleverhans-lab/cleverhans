"""
The Model class and related functionality.
"""
from abc import ABCMeta
import warnings

import tensorflow as tf

from cleverhans import utils_tf


class Model(object):
  """
  An abstract interface for model wrappers that exposes model symbols
  needed for making an attack. This abstraction removes the dependency on
  any specific neural network package (e.g. Keras) from the core
  code of CleverHans. It can also simplify exposing the hidden features of a
  model when a specific package does not directly expose them.
  """
  __metaclass__ = ABCMeta
  O_LOGITS, O_PROBS, O_FEATURES = 'logits probs features'.split()

  def __init__(self, scope=None, nb_classes=None, hparams=None,
               needs_dummy_fprop=False):
    """
    Constructor.
    :param scope: str, the name of model.
    :param nb_classes: integer, the number of classes.
    :param hparams: dict, hyper-parameters for the model.
    :needs_dummy_fprop: bool, if True the model's parameters are not
        created until fprop is called.
    """
    self.scope = scope or self.__class__.__name__
    self.nb_classes = nb_classes
    self.hparams = hparams or {}
    self.needs_dummy_fprop = needs_dummy_fprop

  def __call__(self, *args, **kwargs):
    """
    For compatibility with functions used as model definitions (taking
    an input tensor and returning the tensor giving the output
    of the model on that input).
    """

    warnings.warn("Model.__call__ is deprecated. "
                  "The call is ambiguous as to whether the output should "
                  "be logits or probabilities, and getting the wrong one "
                  "can cause serious problems. "
                  "The output actually is probabilities, which are a very "
                  "dangerous thing to use as part of any interface for "
                  "cleverhans, because softmax probabilities are prone "
                  "to gradient masking."
                  "On or after 2019-04-24, this method will change to raise "
                  "an exception explaining why Model.__call__ should not be "
                  "used.")

    return self.get_probs(*args, **kwargs)

  def get_logits(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the output logits
    (i.e., the values fed as inputs to the softmax layer).
    """
    outputs = self.fprop(x, **kwargs)
    if self.O_LOGITS in outputs:
      return outputs[self.O_LOGITS]
    raise NotImplementedError(str(type(self)) + "must implement `get_logits`"
                              " or must define a " + self.O_LOGITS +
                              " output in `fprop`")

  def get_predicted_class(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the predicted label
    """
    return tf.argmax(self.get_logits(x, **kwargs), axis=1)

  def get_probs(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the output
    probabilities (i.e., the output values produced by the softmax layer).
    """
    d = self.fprop(x, **kwargs)
    if self.O_PROBS in d:
      output = d[self.O_PROBS]
      min_prob = tf.reduce_min(output)
      max_prob = tf.reduce_max(output)
      asserts = [utils_tf.assert_greater_equal(min_prob,
                                               tf.cast(0., min_prob.dtype)),
                 utils_tf.assert_less_equal(max_prob,
                                            tf.cast(1., min_prob.dtype))]
      with tf.control_dependencies(asserts):
        output = tf.identity(output)
      return output
    elif self.O_LOGITS in d:
      return tf.nn.softmax(logits=d[self.O_LOGITS])
    else:
      raise ValueError('Cannot find probs or logits.')

  def fprop(self, x, **kwargs):
    """
    Forward propagation to compute the model outputs.
    :param x: A symbolic representation of the network input
    :return: A dictionary mapping layer names to the symbolic
             representation of their output.
    """
    raise NotImplementedError('`fprop` not implemented.')

  def get_params(self):
    """
    Provides access to the model's parameters.
    :return: A list of all Variables defining the model parameters.
    """

    if hasattr(self, 'params'):
      return list(self.params)

    # Catch eager execution and assert function overload.
    try:
      if tf.executing_eagerly():
        raise NotImplementedError("For Eager execution - get_params "
                                  "must be overridden.")
    except AttributeError:
      pass

    # For graph-based execution
    scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   self.scope + "/")

    if len(scope_vars) == 0:
      self.make_params()
      scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope + "/")
      assert len(scope_vars) > 0

    # Make sure no parameters have been added or removed
    if hasattr(self, "num_params"):
      if self.num_params != len(scope_vars):
        print("Scope: ", self.scope)
        print("Expected " + str(self.num_params) + " variables")
        print("Got " + str(len(scope_vars)))
        for var in scope_vars:
          print("\t" + str(var))
        assert False
    else:
      self.num_params = len(scope_vars)

    return scope_vars

  def make_params(self):
    """
    Create all Variables to be returned later by get_params.
    By default this is a no-op.
    Models that need their fprop to be called for their params to be
    created can set `needs_dummy_fprop=True` in the constructor.
    """

    if self.needs_dummy_fprop:
      if hasattr(self, "_dummy_input"):
        return
      self._dummy_input = self.make_input_placeholder()
      self.fprop(self._dummy_input)

  def get_layer_names(self):
    """Return the list of exposed layers for this model."""
    raise NotImplementedError

  def get_layer(self, x, layer, **kwargs):
    """Return a layer output.
    :param x: tensor, the input to the network.
    :param layer: str, the name of the layer to compute.
    :param **kwargs: dict, extra optional params to pass to self.fprop.
    :return: the content of layer `layer`
    """
    return self.fprop(x, **kwargs)[layer]

  def make_input_placeholder(self):
    """Create and return a placeholder representing an input to the model.

    This method should respect context managers (e.g. "with tf.device")
    and should not just return a reference to a single pre-created
    placeholder.
    """

    raise NotImplementedError(str(type(self)) + " does not implement "
                              "make_input_placeholder")

  def make_label_placeholder(self):
    """Create and return a placeholder representing class labels.

    This method should respect context managers (e.g. "with tf.device")
    and should not just return a reference to a single pre-created
    placeholder.
    """

    raise NotImplementedError(str(type(self)) + " does not implement "
                              "make_label_placeholder")

  def __hash__(self):
    return hash(id(self))

  def __eq__(self, other):
    return self is other


class CallableModelWrapper(Model):
  """A wrapper that turns a callable into a valid Model"""

  def __init__(self, callable_fn, output_layer):
    """
    Wrap a callable function that takes a tensor as input and returns
    a tensor as output with the given layer name.
    :param callable_fn: The callable function taking a tensor and
                        returning a given layer as output.
    :param output_layer: A string of the output layer returned by the
                         function. (Usually either "probs" or "logits".)
    """

    super(CallableModelWrapper, self).__init__()
    self.output_layer = output_layer
    self.callable_fn = callable_fn

  def fprop(self, x, **kwargs):
    output = self.callable_fn(x, **kwargs)

    # Do some sanity checking to reduce the chance that probs are used
    # as logits accidentally or vice versa
    if self.output_layer == 'probs':
      assert output.op.type == "Softmax"
      min_prob = tf.reduce_min(output)
      max_prob = tf.reduce_max(output)
      asserts = [utils_tf.assert_greater_equal(min_prob,
                                               tf.cast(0., min_prob.dtype)),
                 utils_tf.assert_less_equal(max_prob,
                                            tf.cast(1., max_prob.dtype))]
      with tf.control_dependencies(asserts):
        output = tf.identity(output)
    elif self.output_layer == 'logits':
      assert output.op.type != 'Softmax'

    return {self.output_layer: output}

def wrapper_warning():
  """
  Issue a deprecation warning. Used in multiple places that implemented
  attacks by automatically wrapping a user-supplied callable with a
  CallableModelWrapper with output_layer="probs".
  Using "probs" as any part of the attack interface is dangerous.
  We can't just change output_layer to logits because:
  - that would be a silent interface change. We'd have no way of detecting
    code that still means to use probs. Note that we can't just check whether
    the final output op is a softmax---for example, Inception puts a reshape
    after the softmax.
  - automatically wrapping user-supplied callables with output_layer='logits'
    is even worse, see `wrapper_warning_logits`
  Note: this function will be removed at the same time as the code that
  calls it.
  """
  warnings.warn("Passing a callable is deprecated, because using"
                " probabilities is dangerous. It has a high risk "
                " of causing gradient masking due to loss of precision "
                " in the softmax op. Passing a callable rather than a "
                " Model subclass will become an error on or after "
                " 2019-04-24.")

def wrapper_warning_logits():
  """
  Issue a deprecation warning. Used in multiple places that implemented
  attacks by automatically wrapping a user-supplied callable with a
  CallableModelWrapper with output_layer="logits".
  This is dangerous because it is under-the-hood automagic that the user
  may not realize has been invoked for them. If they pass a callable
  that actually outputs probs, the probs will be treated as logits,
  resulting in an incorrect cross-entropy loss and severe gradient
  masking.
  """
  warnings.warn("Passing a callable is deprecated, because it runs the "
                "risk of accidentally using probabilities in the place "
                "of logits. Please switch to passing a Model subclass "
                "so that you clearly specify which values are the logits. "
                "Passing a callable rather than a Model subclass will become "
                "an error on or after 2019-04-24.")


class NoSuchLayerError(ValueError):
  """Raised when a layer that does not exist is requested."""
