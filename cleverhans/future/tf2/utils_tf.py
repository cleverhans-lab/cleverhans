import numpy as np
import tensorflow as tf


def clip_eta(eta, norm, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param norm: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.norm norm ball
  if norm not in [np.inf, 1, 2]:
    raise ValueError('norm must be np.inf, 1, or 2.')
  axis = list(range(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if norm == np.inf:
    eta = tf.clip_by_value(eta, -eps, eps)
  else:
    if norm == 1:
      raise NotImplementedError("")
      # This is not the correct way to project on the L1 norm ball:
      # norm = tf.maximum(avoid_zero_div, reduce_sum(tf.abs(eta), reduc_ind, keepdims=True))
    elif norm == 2:
      # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
      norm = tf.sqrt(
          tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
    factor = tf.minimum(1., tf.math.divide(eps, norm))
    eta = eta * factor
  return eta


def get_or_guess_labels(model_fn, x, y=None, targeted=False):
  """
  Helper function to get the label to use in generating an
  adversarial example for x.
  If 'y' is not None, then use these labels.
  If 'targeted' is True, then assume it's a targeted attack
  and y must be set.
  Otherwise, use the model's prediction as the label and perform an
  untargeted attack
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  """
  if targeted is True and y is None:
    raise ValueError("Must provide y for a targeted attack!")

  preds = model_fn(x)
  nb_classes = preds.shape[-1]

  # labels set by the user
  if y is not None:
    y = np.asarray(y)

    if len(y.shape) == 1:
      # the user provided a list/1D-array
      idx = y.reshape([-1, 1])
      y = np.zeros_like(preds)
      y[:, idx] = 1

    y = tf.cast(y, x.dtype)
    return y, nb_classes

  # must be an untargeted attack
  labels = tf.cast(tf.equal(tf.reduce_max(
      preds, axis=1, keepdims=True), preds), x.dtype)

  return labels, nb_classes


def set_with_mask(x, x_other, mask):
  """Helper function which returns a tensor similar to x with all the values
     of x replaced by x_other where the mask evaluates to true.
  """
  mask = tf.cast(mask, x.dtype)
  ones = tf.ones_like(mask, dtype=x.dtype)
  return x_other * mask + x * (ones - mask)
