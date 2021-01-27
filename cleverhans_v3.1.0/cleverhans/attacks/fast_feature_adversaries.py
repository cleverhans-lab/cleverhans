"""
The FastFeatureAdversaries attack
"""
# pylint: disable=missing-docstring
import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_sum
from cleverhans.model import Model
from cleverhans.utils_tf import clip_eta


class FastFeatureAdversaries(Attack):
  """
  This is a fast implementation of "Feature Adversaries", an attack
  against a target internal representation of a model.
  "Feature adversaries" were originally introduced in (Sabour et al. 2016),
  where the optimization was done using LBFGS.
  Paper link: https://arxiv.org/abs/1511.05122

  This implementation is similar to "Basic Iterative Method"
  (Kurakin et al. 2016) but applied to the internal representations.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a FastFeatureAdversaries instance.
    """
    super(FastFeatureAdversaries, self).__init__(model, sess, dtypestr,
                                                 **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'clip_min', 'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'layer']

    assert isinstance(self.model, Model)

  def parse_params(self,
                   layer=None,
                   eps=0.3,
                   eps_iter=0.05,
                   nb_iter=10,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param layer: (required str) name of the layer to target.
    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # Save attack-specific parameters
    self.layer = layer
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True

  def attack_single_step(self, x, eta, g_feat):
    """
    TensorFlow implementation of the Fast Feature Gradient. This is a
    single step attack similar to Fast Gradient Method that attacks an
    internal representation.

    :param x: the input placeholder
    :param eta: A tensor the same shape as x that holds the perturbation.
    :param g_feat: model's internal tensor for guide
    :return: a tensor for the adversarial example
    """

    adv_x = x + eta
    a_feat = self.model.fprop(adv_x)[self.layer]

    # feat.shape = (batch, c) or (batch, w, h, c)
    axis = list(range(1, len(a_feat.shape)))

    # Compute loss
    # This is a targeted attack, hence the negative sign
    loss = -reduce_sum(tf.square(a_feat - g_feat), axis)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, adv_x)

    # Multiply by constant epsilon
    scaled_signed_grad = self.eps_iter * tf.sign(grad)

    # Add perturbation to original example to obtain adversarial example
    adv_x = adv_x + scaled_signed_grad

    # If clipping is needed,
    # reset all values outside of [clip_min, clip_max]
    if (self.clip_min is not None) and (self.clip_max is not None):
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    adv_x = tf.stop_gradient(adv_x)

    eta = adv_x - x
    eta = clip_eta(eta, self.ord, self.eps)

    return eta

  def generate(self, x, g, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param g: The target value of the symbolic representation
    :param kwargs: See `parse_params`
    """

    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    g_feat = self.model.fprop(g)[self.layer]

    # Initialize loop variables
    eta = tf.random_uniform(
        tf.shape(x), -self.eps, self.eps, dtype=self.tf_dtype)
    eta = clip_eta(eta, self.ord, self.eps)

    def cond(i, _):
      return tf.less(i, self.nb_iter)

    def body(i, e):
      new_eta = self.attack_single_step(x, e, g_feat)
      return i + 1, new_eta

    _, eta = tf.while_loop(cond, body, (tf.zeros([]), eta), back_prop=True,
                           maximum_iterations=self.nb_iter)

    # Define adversarial example (and clip if necessary)
    adv_x = x + eta
    if self.clip_min is not None and self.clip_max is not None:
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    return adv_x
