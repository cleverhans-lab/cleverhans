"""
The Model class and related functionality.
"""
from abc import ABCMeta
import tensorflow as tf


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
    return self.get_probs(*args, **kwargs)

  def get_logits(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the output logits
    (i.e., the values fed as inputs to the softmax layer).
    """
    return self.fprop(x, **kwargs)[self.O_LOGITS]

  def get_probs(self, x, **kwargs):
    """
    :param x: A symbolic representation (Tensor) of the network input
    :return: A symbolic representation (Tensor) of the output
    probabilities (i.e., the output values produced by the softmax layer).
    """
    d = self.fprop(x, **kwargs)
    if self.O_PROBS in d:
      return d[self.O_PROBS]
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
    # Catch eager execution and assert function overload.
    try:
      if tf.executing_eagerly():
        raise NotImplementedError("For Eager execution - get_params "
                                  "must be overridden.")
    except AttributeError:
      pass

    # For Graoh based execution
    scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   self.scope)

    if len(scope_vars) == 0:
      self.make_params()
      scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope)
      assert len(scope_vars) > 0

    # Make sure no parameters have been added or removed
    if hasattr(self, "num_params"):
      assert self.num_params == len(scope_vars)
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
    return {self.output_layer: self.callable_fn(x, **kwargs)}


class NoSuchLayerError(ValueError):

  """Raised when a layer that does not exist is requested."""
