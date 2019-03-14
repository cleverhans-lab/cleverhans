"""
Extra losses

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from cleverhans.loss import Loss, softmax_cross_entropy_with_logits
from cleverhans.loss import WeightedSum, WeightDecay
from cleverhans.utils import safe_zip


class UniformLogitSqueezing(Loss):
  def __init__(self, model,
               **kwargs):
    """Constructor.
    :param model: Model instance, the model on which to apply the loss.
    """
    self.kwargs = kwargs
    Loss.__init__(self, model, locals())

  def fprop(self, x, y, **kwargs):

    kwargs.update(self.kwargs)

    uniform_x = tf.random_uniform(tf.shape(x), -1., 1.)
    logits = self.model.get_logits(x, **kwargs)
    loss = tf.reduce_mean(tf.square(logits))

    return loss


class CrossEntropyALP(Loss):
  """Cross-entropy loss for a multiclass softmax classifier.
  :param model: Model instance, the model on which to apply the loss.
  :param smoothing: float, amount of label smoothing for cross-entropy.
  :param attack: function, given an input x, return an attacked x'.
  :param pass_y: bool, if True pass y to the attack
  """
  def __init__(self, model, attack, smoothing=0., pass_y=False,
               adv_coeff=0.5, attack_params=None,
               alp_coeff=1., stop_clean=False, stop_adv=False,
               **kwargs):
    if smoothing < 0 or smoothing > 1:
      raise ValueError('Smoothing must be in [0, 1]', smoothing)
    self.kwargs = kwargs
    Loss.__init__(self, model, locals(), attack)
    self.smoothing = smoothing
    self.adv_coeff = adv_coeff
    self.pass_y = pass_y
    self.attack_params = attack_params
    self.alp_coeff = alp_coeff
    self.stop_clean = stop_clean
    self.stop_adv = stop_adv

  def fprop(self, x, y, **kwargs):
    kwargs.update(self.kwargs)
    attack_params = copy.copy(self.attack_params)
    if attack_params is None:
        attack_params = {}
    if self.pass_y:
        attack_params['y'] = y
    x = x, self.attack.generate(x, **attack_params)
    coeffs = [1. - self.adv_coeff, self.adv_coeff]

    # Catching RuntimeError: Variable -= value not supported by tf.eager.
    try:
      y -= self.smoothing * (y - 1. / tf.cast(y.shape[-1], y.dtype))
    except RuntimeError:
      y.assign_sub(self.smoothing * (y - 1. / tf.cast(y.shape[-1],
                                                      y.dtype)))

    logits = [self.model.get_logits(x, **kwargs) for x in x]
    loss = sum(
        coeff * tf.reduce_mean(softmax_cross_entropy_with_logits(labels=y,
                                                         logits=logit))
        for coeff, logit in safe_zip(coeffs, logits))
    clean_logits = logits[0]
    adv_logits = logits[1]
    if self.stop_clean:
      clean_logits = tf.stop_gradient(clean_logits)
    if self.stop_adv:
      adv_logits = tf.stop_gradient(adv_logits)
    loss = loss + self.alp_coeff * tf.reduce_mean(tf.square(clean_logits -
                                                            adv_logits))
    return loss
