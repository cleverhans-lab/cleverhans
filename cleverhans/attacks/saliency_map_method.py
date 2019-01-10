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
      assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
      from cleverhans.attacks_tf import jacobian_graph


      # Define Jacobian graph wrt to this input placeholder
      preds = self.model.get_probs(x)
      nb_classes = preds.get_shape().as_list()[-1]
      grads = jacobian_graph(preds, x, nb_classes)

      # Define appropriate graph (targeted / random target labels)
      if self.y_target is not None:

        def jsma_wrap(x_val, y_target):
          return jsma_batch(
              self.sess,
              x,
              preds,
              grads,
              x_val,
              self.theta,
              self.gamma,
              self.clip_min,
              self.clip_max,
              nb_classes,
              y_target=y_target)

        # Attack is targeted, target placeholder will need to be fed
        x_adv = tf.py_func(jsma_wrap, [x, self.y_target],
                           self.tf_dtype)
      else:

        def jsma_wrap(x_val):
          return jsma_batch(
              self.sess,
              x,
              preds,
              grads,
              x_val,
              self.theta,
              self.gamma,
              self.clip_min,
              self.clip_max,
              nb_classes,
              y_target=None)

        # Attack is untargeted, target values will be chosen at random
        x_adv = tf.py_func(jsma_wrap, [x], self.tf_dtype)
        x_adv.set_shape(x.get_shape())

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
  raise NotImplementedError("The jsma_batch function has been removed. Any code that depends on it should be revised.")
