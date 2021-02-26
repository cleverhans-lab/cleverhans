"""
The SparseL1Descent attack.
"""

import warnings
from distutils.version import LooseVersion

import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta, random_lp_vector
from cleverhans.compat import reduce_max, reduce_sum, \
  softmax_cross_entropy_with_logits


class SparseL1Descent(Attack):
  """
  This class implements a variant of Projected Gradient Descent for the l1-norm
  (Tramer and Boneh 2019). The l1-norm case is more tricky than the l-inf and l2
  cases covered by the ProjectedGradientDescent class, because the steepest
  descent direction for the l1-norm is too sparse (it updates a single
  coordinate in the adversarial perturbation in each step). This attack has an
  additional parameter that controls the sparsity of the update step. For
  moderately sparse update steps, the attack vastly outperforms Projected
  Steepest Descent and is competitive with other attacks targeted at the l1-norm
  such as the ElasticNetMethod attack (which is much more computationally
  expensive).
  Paper link (Tramer and Boneh 2019): https://arxiv.org/pdf/1904.13000.pdf

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a SparseL1Descent instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(SparseL1Descent, self).__init__(model, sess=sess,
                                          dtypestr=dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                            'clip_max', 'grad_sparsity')
    self.structural_kwargs = ['nb_iter', 'rand_init', 'clip_grad',
                              'sanity_checks']

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
      eta = random_lp_vector(tf.shape(x), ord=1,
                             eps=tf.cast(self.eps, x.dtype), dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, ord=1, eps=self.eps)
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
      preds_max = tf.reduce_max(model_preds, 1, keepdims=True)
      y = tf.to_float(tf.equal(model_preds, preds_max))
      y = tf.stop_gradient(y)
      targeted = False
      del model_preds

    y_kwarg = 'y_target' if targeted else 'y'

    def cond(i, _):
      """Iterate until requested number of iterations is completed"""
      return tf.less(i, self.nb_iter)

    def body(i, adv_x):
      """Do a projected gradient step"""

      labels, _ = self.get_or_guess_labels(adv_x, {y_kwarg: y})
      logits = self.model.get_logits(adv_x)

      adv_x = sparse_l1_descent(adv_x,
                                logits,
                                y=labels,
                                eps=self.eps_iter,
                                q=self.grad_sparsity,
                                clip_min=self.clip_min,
                                clip_max=self.clip_max,
                                clip_grad=self.clip_grad,
                                targeted=(self.y_target is not None),
                                sanity_checks=self.sanity_checks)

      # Clipping perturbation eta to the l1-ball
      eta = adv_x - x
      eta = clip_eta(eta, ord=1, eps=self.eps)
      adv_x = x + eta

      # Redo the clipping.
      # Subtracting and re-adding eta can add some small numerical error.
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

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

  def parse_params(self,
                   eps=10.0,
                   eps_iter=1.0,
                   nb_iter=20,
                   y=None,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   rand_init=False,
                   clip_grad=False,
                   grad_sparsity=99,
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
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param clip_grad: (optional bool) Ignore gradient components
                      at positions where the input is already at the boundary
                      of the domain, and the update step will get clipped out.
    :param grad_sparsity (optional) Relative sparsity of the gradient update
                         step, in percent. Only gradient values larger
                         than this percentile are retained. This parameter can
                         be a scalar, or a vector of the same length as the
                         input batch dimension.
    :param sanity_checks: bool Insert tf asserts checking values
        (Some tests need to run with no sanity checks because the
         tests intentionally configure the attack strangely)
    """

    # Save attack-specific parameters
    self.eps = eps
    self.rand_init = rand_init
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.clip_grad = clip_grad
    self.grad_sparsity = grad_sparsity

    if isinstance(eps, float) and isinstance(eps_iter, float):
      # If these are both known at compile time, we can check before anything
      # is run. If they are tf, we can't check them yet.
      assert eps_iter <= eps, (eps_iter, eps)

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")

    if self.clip_grad and (self.clip_min is None or self.clip_max is None):
      raise ValueError("Must set clip_min and clip_max if clip_grad is set")

    # The grad_sparsity argument governs the sparsity of the gradient
    # update. It indicates the percentile value above which gradient entries
    # are retained. It can be specified as a scalar or as a 1-dimensional
    # vector of the same size as the input's batch dimension.
    if isinstance(self.grad_sparsity, int) or \
        isinstance(self.grad_sparsity, float):
      if not 0 < self.grad_sparsity < 100:
        raise ValueError("grad_sparsity should be in (0, 100)")
    else:
      self.grad_sparsity = tf.convert_to_tensor(self.grad_sparsity)
      if len(self.grad_sparsity.shape) > 1:
        raise ValueError("grad_sparsity should either be a scalar or a vector")


    self.sanity_checks = sanity_checks

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


def sparse_l1_descent(x,
                      logits,
                      y=None,
                      eps=1.0,
                      q=99,
                      clip_min=None,
                      clip_max=None,
                      clip_grad=False,
                      targeted=False,
                      sanity_checks=True):
  """
  TensorFlow implementation of the Dense L1 Descent Method.
  :param x: the input placeholder
  :param logits: output of model.get_logits
  :param y: (optional) A placeholder for the true labels. If targeted
            is true, then provide the target label. Otherwise, only provide
            this parameter if you'd like to use true labels when crafting
            adversarial samples. Otherwise, model predictions are used as
            labels to avoid the "label leaking" effect (explained in this
            paper: https://arxiv.org/abs/1611.01236). Default is None.
            Labels should be one-hot-encoded.
  :param eps: the epsilon (input variation parameter)
  :param q: the percentile above which gradient values are retained. Either a
            scalar or a vector of same length as the input batch dimension.
  :param clip_min: Minimum float value for adversarial example components
  :param clip_max: Maximum float value for adversarial example components
  :param clip_grad: (optional bool) Ignore gradient components
                    at positions where the input is already at the boundary
                    of the domain, and the update step will get clipped out.
  :param targeted: Is the attack targeted or untargeted? Untargeted, the
                   default, will try to make the label incorrect. Targeted
                   will instead try to move in the direction of being more
                   like y.
  :return: a tensor for the adversarial example
  """

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(clip_min, x.dtype)))

  if clip_max is not None:
    asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

  # Make sure the caller has not passed probs by accident
  assert logits.op.type != 'Softmax'

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    preds_max = reduce_max(logits, 1, keepdims=True)
    y = tf.to_float(tf.equal(logits, preds_max))
    y = tf.stop_gradient(y)
  y = y / reduce_sum(y, 1, keepdims=True)

  # Compute loss
  loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  grad, = tf.gradients(loss, x)

  if clip_grad:
    grad = utils_tf.zero_out_clipped_grads(grad, x, clip_min, clip_max)

  red_ind = list(range(1, len(grad.get_shape())))
  dim = tf.reduce_prod(tf.shape(x)[1:])

  abs_grad = tf.reshape(tf.abs(grad), (-1, dim))

  # if q is a scalar, broadcast it to a vector of same length as the batch dim
  q = tf.cast(tf.broadcast_to(q, tf.shape(x)[0:1]), tf.float32)
  k = tf.cast(tf.floor(q / 100 * tf.cast(dim, tf.float32)), tf.int32)

  # `tf.sort` is much faster than `tf.contrib.distributions.percentile`.
  # For TF <= 1.12, use `tf.nn.top_k` as `tf.sort` is not implemented.
  if LooseVersion(tf.__version__) <= LooseVersion('1.12.0'):
    # `tf.sort` is only available in TF 1.13 onwards
    sorted_grad = -tf.nn.top_k(-abs_grad, k=dim, sorted=True)[0]
  else:
    sorted_grad = tf.sort(abs_grad, axis=-1)

  idx = tf.stack((tf.range(tf.shape(abs_grad)[0]), k), -1)
  percentiles = tf.gather_nd(sorted_grad, idx)
  tied_for_max = tf.greater_equal(abs_grad, tf.expand_dims(percentiles, -1))
  tied_for_max = tf.reshape(tf.cast(tied_for_max, x.dtype), tf.shape(grad))
  num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)

  optimal_perturbation = tf.sign(grad) * tied_for_max / num_ties

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + utils_tf.mul(eps, optimal_perturbation)

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

  if sanity_checks:
    with tf.control_dependencies(asserts):
      adv_x = tf.identity(adv_x)

  return adv_x
