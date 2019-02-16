"""
The ProjectedGradientDescent attack.
"""

import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.attacks.fast_gradient_method import FastGradientMethod
from cleverhans.compat import reduce_max
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta


class ProjectedGradientDescent(Attack):
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param default_rand_init: whether to use random initialization by default
  :param kwargs: passed through to super constructor
  """

  FGM_CLASS = FastGradientMethod

  def __init__(self, model, sess=None, dtypestr='float32',
               default_rand_init=True, **kwargs):
    """
    Create a ProjectedGradientDescent instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(ProjectedGradientDescent, self).__init__(model, sess=sess,
                                                   dtypestr=dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                            'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'sanity_checks']
    self.default_rand_init = default_rand_init

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    asserts = []

    # If a data range was specified, check that the input was in that range
    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(x,
                                                   tf.cast(self.clip_min,
                                                           x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(x,
                                                tf.cast(self.clip_max,
                                                        x.dtype)))

    # Initialize loop variables
    if self.rand_init:
      eta = tf.random_uniform(tf.shape(x),
                              tf.cast(-self.rand_minmax, x.dtype),
                              tf.cast(self.rand_minmax, x.dtype),
                              dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, self.ord, self.eps)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    if self.y_target is not None:
      y = self.y_target
      targeted = True
    elif self.y is not None:
      y = self.y
      targeted = False
    else:
      model_preds = self.model.get_probs(x)
      preds_max = reduce_max(model_preds, 1, keepdims=True)
      y = tf.to_float(tf.equal(model_preds, preds_max))
      y = tf.stop_gradient(y)
      targeted = False
      del model_preds

    y_kwarg = 'y_target' if targeted else 'y'
    fgm_params = {
        'eps': self.eps_iter,
        y_kwarg: y,
        'ord': self.ord,
        'clip_min': self.clip_min,
        'clip_max': self.clip_max
    }
    if self.ord == 1:
      raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                " step for PGD when ord=1, because ord=1 FGM "
                                " changes only one pixel at a time. We need "
                                " to rigorously test a strong ord=1 PGD "
                                "before enabling this feature.")

    # Use getattr() to avoid errors in eager execution attacks
    FGM = self.FGM_CLASS(
        self.model,
        sess=getattr(self, 'sess', None),
        dtypestr=self.dtypestr)

    def cond(i, _):
      """Iterate until requested number of iterations is completed"""
      return tf.less(i, self.nb_iter)

    def body(i, adv_x):
      """Do a projected gradient step"""
      adv_x = FGM.generate(adv_x, **fgm_params)

      # Clipping perturbation eta to self.ord norm ball
      eta = adv_x - x
      eta = clip_eta(eta, self.ord, self.eps)
      adv_x = x + eta

      # Redo the clipping.
      # FGM already did it, but subtracting and re-adding eta can add some
      # small numerical error.
      if self.clip_min is not None or self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      return i + 1, adv_x

    _, adv_x = tf.while_loop(cond, body, (tf.zeros([]), adv_x), back_prop=True,
                             maximum_iterations=self.nb_iter)

    # Asserts run only on CPU.
    # When multi-GPU eval code tries to force all PGD ops onto GPU, this
    # can cause an error.
    common_dtype = tf.float32
    asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps_iter,
                                                      dtype=common_dtype),
                                              tf.cast(self.eps, dtype=common_dtype)))
    if self.ord == np.inf and self.clip_min is not None:
      # The 1e-6 is needed to compensate for numerical error.
      # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5,
      # clip_max=.7
      asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps, x.dtype),
                                                1e-6 + tf.cast(self.clip_max,
                                                               x.dtype)
                                                - tf.cast(self.clip_min,
                                                          x.dtype)))

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

  def parse_params(self,
                   eps=0.3,
                   eps_iter=0.05,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   rand_init=None,
                   rand_minmax=0.3,
                   sanity_checks=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the true labels.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool Insert tf asserts checking values
        (Some tests need to run with no sanity checks because the
         tests intentionally configure the attack strangely)
    """

    # Save attack-specific parameters
    self.eps = eps
    if rand_init is None:
      rand_init = self.default_rand_init
    self.rand_init = rand_init
    if self.rand_init:
      self.rand_minmax = eps
    else:
      self.rand_minmax = 0.
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    if isinstance(eps, float) and isinstance(eps_iter, float):
      # If these are both known at compile time, we can check before anything
      # is run. If they are tf, we can't check them yet.
      assert eps_iter <= eps, (eps_iter, eps)

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")
    self.sanity_checks = sanity_checks

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True
