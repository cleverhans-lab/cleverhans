"""
Wrapper functions for writing code that is compatible with many versions
of TensorFlow.
"""
import warnings
import tensorflow as tf
# The following 2 imports are not used in this module. They are imported so that users of cleverhans.compat can
# get access to device_lib, app, and flags. A pylint bug makes these imports cause errors when using python3+tf1.8.
# Doing the sanitized import here once makes it possible to do "from cleverhans.compat import flags" throughout the
# library without needing to repeat the pylint boilerplate.
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module,unused-import
from tensorflow.python.platform import app, flags # pylint: disable=no-name-in-module,unused-import
from tensorflow import reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_prod, reduce_any

def _wrap(f):
  """
  Wraps a callable `f` in a function that warns that the function is deprecated.
  """
  def wrapper(*args, **kwargs):
    """
    Issues a deprecation warning and passes through the arguments.
    """
    warnings.warn(str(f) + " is deprecated. Switch to calling the equivalent function in tensorflow. "
                  " This function was originally needed as a compatibility layer for old versions of tensorflow, "
                  " but support for those versions has now been dropped.")
    return f(*args, **kwargs)
  return wrapper

reduce_sum = _wrap(reduce_sum)
reduce_max = _wrap(reduce_max)
reduce_min = _wrap(reduce_min)
reduce_mean = _wrap(reduce_mean)
reduce_prod = _wrap(reduce_prod)
reduce_any = _wrap(reduce_any)

def reduce_function(op_func, input_tensor, axis=None, keepdims=None,
                    name=None, reduction_indices=None):
  """
  This function used to be needed to support tf 1.4 and early, but support for tf 1.4 and earlier is now dropped.
  :param op_func: expects the function to handle eg: tf.reduce_sum.
  :param input_tensor: The tensor to reduce. Should have numeric type.
  :param axis: The dimensions to reduce. If None (the default),
          reduces all dimensions. Must be in the range
          [-rank(input_tensor), rank(input_tensor)).
  :param keepdims: If true, retains reduced dimensions with length 1.
  :param name: A name for the operation (optional).
  :param reduction_indices: The old (deprecated) name for axis.
  :return: outputs same value as op_func.
  """

  warnings.warn("`reduce_function` is deprecated and may be removed on or after 2019-09-08.")

  out = op_func(input_tensor, axis=axis, keepdims=keepdims, name=name, reduction_indices=reduction_indices)

  return out

def softmax_cross_entropy_with_logits(sentinel=None,
                                      labels=None,
                                      logits=None,
                                      dim=-1):
  """
  Wrapper around tf.nn.softmax_cross_entropy_with_logits_v2 to handle
  deprecated warning
  """
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    name = "softmax_cross_entropy_with_logits"
    raise ValueError("Only call `%s` with "
                     "named arguments (labels=..., logits=..., ...)"
                     % name)
  if labels is None or logits is None:
    raise ValueError("Both labels and logits must be provided.")

  try:
    f = tf.nn.softmax_cross_entropy_with_logits_v2
  except AttributeError:
    raise RuntimeError("This version of TensorFlow is no longer supported. See cleverhans/README.md")

  labels = tf.stop_gradient(labels)
  loss = f(labels=labels, logits=logits, dim=dim)

  return loss
