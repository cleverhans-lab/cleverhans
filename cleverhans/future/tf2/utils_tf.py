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


def random_exponential(shape, rate=1.0, dtype=tf.float32, seed=None):
  """
  Helper function to sample from the exponential distribution, which is not
  included in core TensorFlow.
  
  shape: shape of the sampled tensor.
  :rate: (optional) rate parameter of the exponential distribution, defaults to 1.0.
  :dtype: (optional) data type of the sempled tensor, defaults to tf.float32.
  :seed: (optional) custom seed to be used for sampling.
  """
  return tf.random.gamma(shape, alpha=1, beta=1. / rate, dtype=dtype, seed=seed)


def random_laplace(shape, loc=0.0, scale=1.0, dtype=tf.float32, seed=None):
  """
  Helper function to sample from the Laplace distribution, which is not
  included in core TensorFlow.
  
  :shape: shape of the sampled tensor.
  :loc: (optional) mean of the laplace distribution, defaults to 0.0.
  :scale: (optional) scale parameter of the laplace diustribution, defaults to 1.0.
  :dtype: (optional) data type of the sempled tensor, defaults to tf.float32.
  :seed: (optional) custom seed to be used for sampling.
  """
  z1 = random_exponential(shape, 1. / scale, dtype=dtype, seed=seed)
  z2 = random_exponential(shape, 1. / scale, dtype=dtype, seed=seed)
  return z1 - z2 + loc

def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
  """
  Helper function to generate uniformly random vectors from a norm ball of
  radius epsilon.
  :param shape: Output shape of the random sample. The shape is expected to be
                of the form `(n, d1, d2, ..., dn)` where `n` is the number of
                i.i.d. samples that will be drawn from a norm ball of dimension
                `d1*d1*...*dn`.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, radius of the norm ball.
  :param dtype: (optional) type of the tensor.
  :param seed: (optional) integer.
  """
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')

  if ord == np.inf:
    r = tf.random.uniform(shape, -eps, eps, dtype=dtype, seed=seed)
  else:

    # For ord=1 and ord=2, we use the generic technique from
    # (Calafiore et al. 1998) to sample uniformly from a norm ball.
    # Paper link (Calafiore et al. 1998):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=758215&tag=1
    # We first sample from the surface of the norm ball, and then scale by
    # a factor `w^(1/d)` where `w~U[0,1]` is a standard uniform random variable
    # and `d` is the dimension of the ball. In high dimensions, this is roughly
    # equivalent to sampling from the surface of the ball.

    dim = tf.reduce_prod(shape[1:])

    if ord == 1:
      x = random_laplace((shape[0], dim), loc=1.0, scale=1.0, dtype=dtype,
                         seed=seed)
      norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
    elif ord == 2:
      x = tf.random.normal((shape[0], dim), dtype=dtype, seed=seed)
      norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
    else:
      raise ValueError('ord must be np.inf, 1, or 2.')

    w = tf.pow(tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
               1.0 / tf.cast(dim, dtype))
    r = eps * tf.reshape(w * x / norm, shape)

  return r

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
    # inefficient when y is a tensor, but this function only get called once
    y = np.asarray(y)

    if len(y.shape) == 1:
      # the user provided categorical encoding
      y = tf.one_hot(y, nb_classes)

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
