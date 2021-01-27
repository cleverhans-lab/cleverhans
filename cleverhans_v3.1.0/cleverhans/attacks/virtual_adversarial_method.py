"""The VirtualAdversarialMethod attack

"""

import warnings

import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.model import wrapper_warning_logits
from cleverhans import utils_tf

tf_dtype = tf.as_dtype('float32')

class VirtualAdversarialMethod(Attack):
  """
  This attack was originally proposed by Miyato et al. (2016) and was used
  for virtual adversarial training.
  Paper link: https://arxiv.org/abs/1507.00677

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
    if not isinstance(model, Model):
      wrapper_warning_logits()
      model = CallableModelWrapper(model, 'logits')

    super(VirtualAdversarialMethod, self).__init__(model, sess, dtypestr,
                                                   **kwargs)

    self.feedable_kwargs = ('eps', 'xi', 'clip_min', 'clip_max')
    self.structural_kwargs = ['num_iterations']

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    return vatm(
        self.model,
        x,
        self.model.get_logits(x),
        eps=self.eps,
        num_iterations=self.num_iterations,
        xi=self.xi,
        clip_min=self.clip_min,
        clip_max=self.clip_max)

  def parse_params(self,
                   eps=2.0,
                   nb_iter=None,
                   xi=1e-6,
                   clip_min=None,
                   clip_max=None,
                   num_iterations=None,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float )the epsilon (input variation parameter)
    :param nb_iter: (optional) the number of iterations
      Defaults to 1 if not specified
    :param xi: (optional float) the finite difference parameter
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param num_iterations: Deprecated alias for `nb_iter`
    """
    # Save attack-specific parameters
    self.eps = eps
    if num_iterations is not None:
      warnings.warn("`num_iterations` is deprecated. Switch to `nb_iter`."
                    " The old name will be removed on or after 2019-04-26.")
      # Note: when we remove the deprecated alias, we can put the default
      # value of 1 for nb_iter back in the method signature
      assert nb_iter is None
      nb_iter = num_iterations
    del num_iterations
    if nb_iter is None:
      nb_iter = 1
    self.num_iterations = nb_iter
    self.xi = xi
    self.clip_min = clip_min
    self.clip_max = clip_max
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")
    return True


def vatm(model,
         x,
         logits,
         eps,
         num_iterations=1,
         xi=1e-6,
         clip_min=None,
         clip_max=None,
         scope=None):
  """
  Tensorflow implementation of the perturbation method used for virtual
  adversarial training: https://arxiv.org/abs/1507.00677
  :param model: the model which returns the network unnormalized logits
  :param x: the input placeholder
  :param logits: the model's unnormalized output tensor (the input to
                 the softmax layer)
  :param eps: the epsilon (input variation parameter)
  :param num_iterations: the number of iterations
  :param xi: the finite difference parameter
  :param clip_min: optional parameter that can be used to set a minimum
                  value for components of the example returned
  :param clip_max: optional parameter that can be used to set a maximum
                  value for components of the example returned
  :param seed: the seed for random generator
  :return: a tensor for the adversarial example
  """
  with tf.name_scope(scope, "virtual_adversarial_perturbation"):
    d = tf.random_normal(tf.shape(x), dtype=tf_dtype)
    for _ in range(num_iterations):
      d = xi * utils_tf.l2_batch_normalize(d)
      logits_d = model.get_logits(x + d)
      kl = utils_tf.kl_with_logits(logits, logits_d)
      Hd = tf.gradients(kl, d)[0]
      d = tf.stop_gradient(Hd)
    d = eps * utils_tf.l2_batch_normalize(d)
    adv_x = x + d
    if (clip_min is not None) and (clip_max is not None):
      adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    return adv_x
