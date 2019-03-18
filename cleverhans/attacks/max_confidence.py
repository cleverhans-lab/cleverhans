"""The MaxConfidence attack.
"""
import warnings

import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.attacks.projected_gradient_descent import ProjectedGradientDescent
from cleverhans.model import Model


class MaxConfidence(Attack):
  """
  The MaxConfidence attack.

  An attack designed for use against models that use confidence thresholding
  as a defense.
  If the underlying optimizer is optimal, this attack procedure gives the
  optimal failure rate for every confidence threshold t > 0.5.

  Publication: https://openreview.net/forum?id=H1g0piA9tQ

  :param model: cleverhans.model.Model
  :param sess: optional tf.session.Session
  :param base_attacker: cleverhans.attacks.Attack
  """

  def __init__(self, model, sess=None, base_attacker=None):
    if not isinstance(model, Model):
      raise TypeError("Model must be cleverhans.model.Model, got " +
                      str(type(model)))

    super(MaxConfidence, self).__init__(model, sess)
    if base_attacker is None:
      self.base_attacker = ProjectedGradientDescent(model, sess=sess)
    else:
      self.base_attacker = base_attacker
    self.structural_kwargs = self.base_attacker.structural_kwargs
    self.feedable_kwargs = self.base_attacker.feedable_kwargs

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: Keyword arguments for the base attacker
    """

    assert self.parse_params(**kwargs)
    labels, _nb_classes = self.get_or_guess_labels(x, kwargs)
    adv_x = self.attack(x, labels)

    return adv_x

  def parse_params(self, y=None, nb_classes=10, **kwargs):
    self.y = y
    self.nb_classes = nb_classes
    self.params = kwargs
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")
    return True

  def attack(self, x, true_y):
    """
    Runs the untargeted attack.
    :param x: The input
    :param true_y: The correct label for `x`. This attack aims to produce misclassification.
    """
    adv_x_cls = []
    prob_cls = []
    m = tf.shape(x)[0]
    true_y_idx = tf.argmax(true_y, axis=1)

    expanded_x = tf.concat([x] * self.nb_classes, axis=0)
    target_ys = [tf.to_float(tf.one_hot(tf.ones(m, dtype=tf.int32) * cls,
                                        self.nb_classes))
                 for cls in range(self.nb_classes)]
    target_y = tf.concat(target_ys, axis=0)
    adv_x_cls = self.attack_class(expanded_x, target_y)
    expanded_all_probs = self.model.get_probs(adv_x_cls)

    adv_x_list = tf.split(adv_x_cls, self.nb_classes)
    all_probs_list = tf.split(expanded_all_probs, self.nb_classes)

    for cls in range(self.nb_classes):
      target_y = target_ys[cls]
      all_probs = all_probs_list[cls]
      # We don't actually care whether we hit the target class.
      # We care about the probability of the most likely wrong class
      cur_prob_cls = tf.reduce_max(all_probs - true_y, axis=1)
      # Knock out examples that are correctly classified.
      # This is not needed to be optimal for t >= 0.5, but may as well do it
      # to get better failure rate at lower thresholds.
      chosen_cls = tf.argmax(all_probs, axis=1)
      eligible = tf.to_float(tf.not_equal(true_y_idx, chosen_cls))
      cur_prob_cls = cur_prob_cls * eligible
      prob_cls.append(cur_prob_cls)

    probs = tf.concat([tf.expand_dims(e, 1) for e in prob_cls], axis=1)
    # Don't need to censor here because we knocked out the true class above
    # probs = probs - true_y
    most_confident = tf.argmax(probs, axis=1)
    fused_mask = tf.one_hot(most_confident, self.nb_classes)
    masks = tf.split(fused_mask, num_or_size_splits=self.nb_classes, axis=1)
    shape = [m] + [1] * (len(x.get_shape()) - 1)
    reshaped_masks = [tf.reshape(mask, shape) for mask in masks]
    out = sum(adv_x * rmask for adv_x,
              rmask in zip(adv_x_list, reshaped_masks))
    return out

  def attack_class(self, x, target_y):
    """
    Run the attack on a specific target class.
    :param x: tf Tensor. The input example.
    :param target_y: tf Tensor. The attacker's desired target class.
    Returns:
      A targeted adversarial example, intended to be classified as the target class.
    """
    adv = self.base_attacker.generate(x, y_target=target_y, **self.params)
    return adv
