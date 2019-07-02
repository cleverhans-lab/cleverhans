import jax.numpy as np


def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def clip_eta(eta, norm, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param norm: Order of the norm (mimics Numpy).
              Possible values: np.inf or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.norm norm ball
  if norm not in [np.inf, 2]:
    raise ValueError('norm must be np.inf or 2.')

  axis = list(range(1, len(eta.shape)))
  avoid_zero_div = 1e-12
  if norm == np.inf:
    eta = np.clip(eta, a_min=-eps, a_max=eps)
  elif norm == 2:
    # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
    norm = np.sqrt(np.maximum(avoid_zero_div, np.sum(np.square(eta), axis=axis, keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
    factor = np.minimum(1., np.divide(eps, norm))
    eta = eta * factor
  return eta
