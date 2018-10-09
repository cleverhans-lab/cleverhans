"""
The MaxConfidence attack.

An attack designed for use against models that use confidence thresholding
as a defense.
If the underlying optimizer is optimal, this attack procedure gives the
optimal failure rate for every confidence threshold t > 0.5.

Publication: https://openreview.net/forum?id=H1g0piA9tQ
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from cleverhans.attacks import Attack
from cleverhans.model import Model
from cleverhans.attacks import ProjectedGradientDescent


class MaxConfidence(Attack):
  """
  Initializes with noise and runs iterative optimizer on all classes
  separately, then chooses the wrong class that had highest confidence.

  Publication: https://openreview.net/forum?id=H1g0piA9tQ
  """

  def __init__(self, model, sess=None):
    if not isinstance(model, Model):
      raise TypeError("Model must be cleverhans.model.Model, got " +
                      str(type(model)))

    super(MaxConfidence, self).__init__(model, sess)
    self.base_attacker = ProjectedGradientDescent(model, sess=sess)
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
    return True

  def attack(self, x, true_y):
    adv_x_cls = []
    prob_cls = []
    m = tf.shape(x)[0]
    true_y_idx = tf.argmax(true_y, axis=1)

    expanded_x = tf.concat([x] * self.nb_classes, axis=0)
    target_ys = [tf.to_float(tf.one_hot(tf.ones(m, dtype=tf.int32) * cls, self.nb_classes))
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
    adv = self.base_attacker.generate(x, y_target=target_y, **self.params)
    return adv
