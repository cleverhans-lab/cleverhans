"""The SalienceMapMethod attack
"""
import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack


class SaliencyMapMethod(Attack):
  """
  The Jacobian-based Saliency Map Method (Papernot et al. 2016).
  Paper link: https://arxiv.org/pdf/1511.07528.pdf

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor

  :note: When not using symbolic implementation in `generate`, `sess` should
         be provided
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a SaliencyMapMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(SaliencyMapMethod, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('y_target',)
    self.structural_kwargs = [
        'theta', 'gamma', 'clip_max', 'clip_min', 'symbolic_impl'
    ]

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    if self.symbolic_impl:
      from cleverhans.attacks_tf import jsma_symbolic

      # Create random targets if y_target not provided
      if self.y_target is None:
        from random import randint

        def random_targets(gt):
          result = gt.copy()
          nb_s = gt.shape[0]
          nb_classes = gt.shape[1]

          for i in range(nb_s):
            result[i, :] = np.roll(result[i, :],
                                   randint(1, nb_classes - 1))

          return result

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.y_target = tf.py_func(random_targets, [labels],
                                   self.tf_dtype)
        self.y_target.set_shape([None, nb_classes])

      x_adv = jsma_symbolic(
          x,
          model=self.model,
          y_target=self.y_target,
          theta=self.theta,
          gamma=self.gamma,
          clip_min=self.clip_min,
          clip_max=self.clip_max)
    else:
      raise NotImplementedError("The jsma_batch function has been removed."
                                " The symbolic_impl argument to SaliencyMapMethod will be removed"
                                " on 2019-07-18 or after. Any code that depends on the non-symbolic"
                                " implementation of the JSMA should be revised. Consider using"
                                " SaliencyMapMethod.generate_np() instead.")

    return x_adv

  def parse_params(self,
                   theta=1.,
                   gamma=1.,
                   clip_min=0.,
                   clip_max=1.,
                   y_target=None,
                   symbolic_impl=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param theta: (optional float) Perturbation introduced to modified
                  components (can be positive or negative)
    :param gamma: (optional float) Maximum percentage of perturbed features
    :param clip_min: (optional float) Minimum component value for clipping
    :param clip_max: (optional float) Maximum component value for clipping
    :param y_target: (optional) Target tensor if the attack is targeted
    """
    self.theta = theta
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.y_target = y_target
    self.symbolic_impl = symbolic_impl

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


def jsma_batch(*args, **kwargs):
  raise NotImplementedError(
      "The jsma_batch function has been removed. Any code that depends on it should be revised.")
