import numpy as np
import tensorflow as tf


def clip_eta(eta, ord, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.ord norm ball
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')
  axis = list(range(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    eta = tf.clip_by_value(eta, -eps, eps)
  else:
    if ord == 1:
      raise NotImplementedError("")
      # This is not the correct way to project on the L1 norm ball:
      # norm = tf.maximum(avoid_zero_div, reduce_sum(tf.abs(eta), reduc_ind, keepdims=True))
    elif ord == 2:
      # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
      norm = tf.sqrt(
        tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
    factor = tf.minimum(1., tf.math.divide(eps, norm))
    eta = eta * factor
  return eta
