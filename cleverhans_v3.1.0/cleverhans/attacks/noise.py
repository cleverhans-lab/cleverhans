"""The Noise attack

"""
import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack


class Noise(Attack):
  """
  A weak attack that just picks a random point in the attacker's action space.
  When combined with an attack bundling function, this can be used to implement
  random search.

  References:
  https://arxiv.org/abs/1802.00420 recommends random search to help identify
    gradient masking.
  https://openreview.net/forum?id=H1g0piA9tQ recommends using noise as part
    of an attack bundling recipe combining many different optimizers to yield
    a stronger optimizer.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32',
               **kwargs):

    super(Noise, self).__init__(model, sess=sess, dtypestr=dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'clip_min', 'clip_max')
    self.structural_kwargs = ['ord']

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    if self.ord != np.inf:
      raise NotImplementedError(self.ord)
    eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps,
                            dtype=self.tf_dtype)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      assert self.clip_min is not None and self.clip_max is not None
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    return adv_x

  def parse_params(self,
                   eps=0.3,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # Save attack-specific parameters
    self.eps = eps
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf]:
      raise ValueError("Norm order must be np.inf")
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True
