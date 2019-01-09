"""The SPSA attack
"""
import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.attacks_tf import SPSAAdam, margin_logit_loss, TensorAdam
from cleverhans.compat import reduce_mean
from cleverhans.model import Model
from cleverhans import utils_tf

class SPSA(Attack):
  """
  This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
  (Uesato et al. 2018). SPSA is a gradient-free optimization method, which
  is useful when the model is non-differentiable, or more generally, the
  gradients do not point in useful directions.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  DEFAULT_SPSA_SAMPLES = 128
  DEFAULT_SPSA_ITERS = 1
  DEFAULT_DELTA = 0.01
  DEFAULT_LEARNING_RATE = 0.01

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(SPSA, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('eps', 'clip_min', 'clip_max', 'y', 'y_target')
    self.structural_kwargs = [
        'nb_iter',
        'spsa_samples',
        'spsa_iters',
        'early_stop_loss_threshold',
        'is_debug',
        'is_targeted',
    ]

    assert isinstance(self.model, Model)

  def generate(self,
               x,
               y=None,
               y_target=None,
               eps=None,
               clip_min=None,
               clip_max=None,
               nb_iter=None,
               is_targeted=None,
               early_stop_loss_threshold=None,
               learning_rate=DEFAULT_LEARNING_RATE,
               delta=DEFAULT_DELTA,
               spsa_samples=DEFAULT_SPSA_SAMPLES,
               batch_size=None,
               spsa_iters=DEFAULT_SPSA_ITERS,
               is_debug=False,
               epsilon=None,
               num_steps=None):
    """
    Generate symbolic graph for adversarial examples.

    :param x: The model's symbolic inputs. Must be a batch of size 1.
    :param y: A Tensor or None. The index of the correct label.
    :param y_target: A Tensor or None. The index of the target label in a
                     targeted attack.
    :param eps: The size of the maximum perturbation, measured in the
                L-infinity norm.
    :param clip_min: If specified, the minimum input value
    :param clip_max: If specified, the maximum input value
    :param nb_iter: The number of optimization steps.
    :param early_stop_loss_threshold: A float or None. If specified, the
                                      attack will end as soon as the loss
                                      is below `early_stop_loss_threshold`.
    :param learning_rate: Learning rate of ADAM optimizer.
    :param delta: Perturbation size used for SPSA approximation.
    :param spsa_samples: Number of inputs to evaluate at a single time.
                       The true batch size (the number of evaluated
                       inputs for each update) is `spsa_samples *
                       spsa_iters`
    :param batch_size: Deprecated param that is an alias for spsa_samples
    :param spsa_iters: Number of model evaluations before performing an
                       update, where each evaluation is on `spsa_samples`
                       different inputs.
    :param is_debug: If True, print the adversarial loss after each update.
    :param epsilon: Deprecated alias for `eps`
    :param num_steps: Deprecated alias for `nb_iter`.
    :param is_targeted: Deprecated argument. Ignored.
    """

    if epsilon is not None:
      if eps is not None:
        raise ValueError("Should not specify both eps and its deprecated "
                         "alias, epsilon")
      warnings.warn("`epsilon` is deprecated. Switch to `eps`. `epsilon` may "
                    "be removed on or after 2019-04-15.")
      eps = epsilon
    del epsilon

    if num_steps is not None:
      if nb_iter is not None:
        raise ValueError("Should not specify both nb_iter and its deprecated "
                         "alias, num_steps")
      warnings.warn("`num_steps` is deprecated. Switch to `nb_iter`. "
                    "`num_steps` may be removed on or after 2019-04-15.")
      nb_iter = num_steps
    del num_steps
    assert nb_iter is not None

    if (y is not None) + (y_target is not None) != 1:
      raise ValueError("Must specify exactly one of y (untargeted attack, "
                       "cause the input not to be classified as this true "
                       "label) and y_target (targeted attack, cause the "
                       "input to be classified as this target label).")

    if is_targeted is not None:
      warnings.warn("`is_targeted` is deprecated. Simply do not specify it."
                    " It may become an error to specify it on or after "
                    "2019-04-15.")
      assert is_targeted == y_target is not None

    is_targeted = y_target is not None

    if x.get_shape().as_list()[0] is None:
      check_batch = utils_tf.assert_equal(tf.shape(x)[0], 1)
      with tf.control_dependencies([check_batch]):
        x = tf.identity(x)
    elif x.get_shape().as_list()[0] != 1:
      raise ValueError("For SPSA, input tensor x must have batch_size of 1.")

    if batch_size is not None:
      warnings.warn(
          'The "batch_size" argument to SPSA is deprecated, and will '
          'be removed on 2019-03-17. '
          'Please use spsa_samples instead.')
      spsa_samples = batch_size

    optimizer = SPSAAdam(
        lr=learning_rate,
        delta=delta,
        num_samples=spsa_samples,
        num_iters=spsa_iters)

    def loss_fn(x, label):
      logits = self.model.get_logits(x)
      loss_multiplier = 1 if is_targeted else -1
      return loss_multiplier * margin_logit_loss(
          logits, label,
          nb_classes=self.model.nb_classes or logits.get_shape()[-1])

    y_attack = y_target if is_targeted else y
    adv_x = projected_optimization(
        loss_fn,
        x,
        y_attack,
        eps,
        num_steps=nb_iter,
        optimizer=optimizer,
        early_stop_loss_threshold=early_stop_loss_threshold,
        is_debug=is_debug,
        clip_min=clip_min,
        clip_max=clip_max
    )
    return adv_x

  def generate_np(self, x_val, **kwargs):
    if "epsilon" in kwargs:
      warnings.warn("Using deprecated argument: see `generate`")
      assert "eps" not in kwargs
      kwargs["eps"] = kwargs["epsilon"]
      del kwargs["epsilon"]
    assert "eps" in kwargs

    if "num_steps" in kwargs:
      warnings.warn("Using deprecated argument: see `generate`")
      assert "nb_iter" not in kwargs
      kwargs["nb_iter"] = kwargs["num_steps"]
      del kwargs["num_steps"]

    if 'y' in kwargs and kwargs['y'] is not None:
      assert kwargs['y'].dtype == np.int32
    if 'y_target' in kwargs and kwargs['y_target'] is not None:
      assert kwargs['y_target'].dtype == np.int32

    # Call self.generate() sequentially for each image in the batch
    x_adv = []
    batch_size = x_val.shape[0]
    y = kwargs.pop('y', [None] * batch_size)
    assert len(x_val) == len(y), '# of images and labels should match'
    for x_single, y_single in zip(x_val, y):
      x = np.expand_dims(x_single, axis=0)
      adv_img = super(SPSA, self).generate_np(x, y=y_single, **kwargs)
      x_adv.append(adv_img)
    return np.concatenate(x_adv, axis=0)

def _project_perturbation(perturbation, epsilon, input_image, clip_min=None,
                          clip_max=None):
  """Project `perturbation` onto L-infinity ball of radius `epsilon`.
  Also project into hypercube such that the resulting adversarial example
  is between clip_min and clip_max, if applicable.
  """

  if clip_min is None or clip_max is None:
    raise NotImplementedError("_project_perturbation currently has clipping "
                              "hard-coded in.")

  # Ensure inputs are in the correct range
  with tf.control_dependencies([
      utils_tf.assert_less_equal(input_image,
                                 tf.cast(clip_max, input_image.dtype)),
      utils_tf.assert_greater_equal(input_image,
                                    tf.cast(clip_min, input_image.dtype))
  ]):
    clipped_perturbation = utils_tf.clip_by_value(
        perturbation, -epsilon, epsilon)
    new_image = utils_tf.clip_by_value(
        input_image + clipped_perturbation, clip_min, clip_max)
    return new_image - input_image


def projected_optimization(loss_fn,
                           input_image,
                           label,
                           epsilon,
                           num_steps,
                           clip_min=None,
                           clip_max=None,
                           optimizer=TensorAdam(),
                           project_perturbation=_project_perturbation,
                           early_stop_loss_threshold=None,
                           is_debug=False):
  """Generic projected optimization, generalized to work with approximate
  gradients. Used for e.g. the SPSA attack.

  Args:
    :param loss_fn: A callable which takes `input_image` and `label` as
                    arguments, and returns a batch of loss values. Same
                    interface as TensorOptimizer.
    :param input_image: Tensor, a batch of images
    :param label: Tensor, a batch of labels
    :param epsilon: float, the L-infinity norm of the maximum allowable
                    perturbation
    :param num_steps: int, the number of steps of gradient descent
    :param clip_min: float, minimum pixel value
    :param clip_max: float, maximum pixel value
    :param optimizer: A `TensorOptimizer` object
    :param project_perturbation: A function, which will be used to enforce
                                 some constraint. It should have the same
                                 signature as `_project_perturbation`.
    :param early_stop_loss_threshold: A float or None. If specified, the attack will end if the loss is below
       `early_stop_loss_threshold`.
        Enabling this option can have several different effects:
          - Setting the threshold to 0. guarantees that if a successful attack is found, it is returned.
            This increases the attack success rate, because without early stopping the optimizer can accidentally
            bounce back to a point where the attack fails.
          - Early stopping can make the attack run faster because it may run for fewer steps.
          - Early stopping can make the attack run slower because the loss must be calculated at each step.
            The loss is not calculated as part of the normal SPSA optimization procedure.
            For most reasonable choices of hyperparameters, early stopping makes the attack much faster because
            it decreases the number of steps dramatically.
    :param is_debug: A bool. If True, print debug info for attack progress.

  Returns:
    adversarial version of `input_image`, with L-infinity difference less than
      epsilon, which tries to minimize loss_fn.

  Note that this function is not intended as an Attack by itself. Rather, it
  is designed as a helper function which you can use to write your own attack
  methods. The method uses a tf.while_loop to optimize a loss function in
  a single sess.run() call.
  """
  assert num_steps is not None
  if is_debug:
    with tf.device("/cpu:0"):
      input_image = tf.Print(
          input_image, [],
          "Starting PGD attack with epsilon: %s" % epsilon)

  init_perturbation = tf.random_uniform(
      tf.shape(input_image),
      minval=tf.cast(-epsilon, input_image.dtype),
      maxval=tf.cast(epsilon, input_image.dtype),
      dtype=input_image.dtype)
  init_perturbation = project_perturbation(init_perturbation, epsilon,
                                           input_image, clip_min=clip_min,
                                           clip_max=clip_max)
  init_optim_state = optimizer.init_state([init_perturbation])
  nest = tf.contrib.framework.nest

  def loop_body(i, perturbation, flat_optim_state):
    """Update perturbation to input image."""
    optim_state = nest.pack_sequence_as(
        structure=init_optim_state, flat_sequence=flat_optim_state)

    def wrapped_loss_fn(x):
      return loss_fn(input_image + x, label)

    new_perturbation_list, new_optim_state = optimizer.minimize(
        wrapped_loss_fn, [perturbation], optim_state)
    projected_perturbation = project_perturbation(new_perturbation_list[0],
                                                  epsilon, input_image,
                                                  clip_min=clip_min,
                                                  clip_max=clip_max)

    # Be careful with this bool. A value of 0. is a valid threshold but evaluates to False, so we must explicitly
    # check whether the value is None.
    early_stop = early_stop_loss_threshold is not None
    compute_loss = is_debug or early_stop
    # Don't waste time building the loss graph if we're not going to use it
    if compute_loss:
      # NOTE: this step is not actually redundant with the optimizer step.
      # SPSA calculates the loss at randomly perturbed points but doesn't calculate the loss at the current point.
      loss = reduce_mean(wrapped_loss_fn(projected_perturbation), axis=0)

      if is_debug:
        with tf.device("/cpu:0"):
          loss = tf.Print(loss, [loss], "Total batch loss")

      if early_stop:
        i = tf.cond(tf.less(loss, early_stop_loss_threshold), lambda: float(num_steps), lambda: i)

    return i + 1, projected_perturbation, nest.flatten(new_optim_state)

  def cond(i, *_):
    return tf.less(i, num_steps)

  flat_init_optim_state = nest.flatten(init_optim_state)
  _, final_perturbation, _ = tf.while_loop(
      cond,
      loop_body,
      loop_vars=(tf.constant(0.), init_perturbation, flat_init_optim_state),
      parallel_iterations=1,
      back_prop=False,
      maximum_iterations=num_steps)
  if project_perturbation is _project_perturbation:
    # TODO: this assert looks totally wrong.
    # Not bothering to fix it now because it's only an assert.
    # 1) Multiplying by 1.1 gives a huge margin of error. This should probably
    #    take the difference and allow a tolerance of 1e-6 or something like
    #    that.
    # 2) I think it should probably check the *absolute value* of
    # final_perturbation
    perturbation_max = epsilon * 1.1
    check_diff = utils_tf.assert_less_equal(
        final_perturbation,
        tf.cast(perturbation_max, final_perturbation.dtype),
        message="final_perturbation must change no pixel by more than "
                "%s" % perturbation_max)
  else:
    # TODO: let caller pass in a check_diff function as well as
    # project_perturbation
    check_diff = tf.no_op()

  if clip_min is None or clip_max is None:
    raise NotImplementedError("This function only supports clipping for now")
  check_range = [utils_tf.assert_less_equal(input_image,
                                            tf.cast(clip_max,
                                                    input_image.dtype)),
                 utils_tf.assert_greater_equal(input_image,
                                               tf.cast(clip_min,
                                                       input_image.dtype))]

  with tf.control_dependencies([check_diff] + check_range):
    adversarial_image = input_image + final_perturbation
  return tf.stop_gradient(adversarial_image)
