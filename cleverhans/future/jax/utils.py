import jax.numpy as np


def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def clip_eta(eta, ord, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.ord norm ball
  if ord not in [np.inf]:
    raise ValueError('ord must be np.inf.')
  axis = list(range(1, len(eta.shape)))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    eta = np.clip(eta, a_min=-eps, a_max=eps)
  else:
    raise ValueError('ord must be np.inf.')
  return eta