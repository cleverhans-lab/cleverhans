# pylint: disable=missing-docstring

import tensorflow as tf

tf_dtype = tf.as_dtype('float32')


def spsa(model_fn, x, y, eps, nb_iter, clip_min=None, clip_max=None, targeted=False,
         early_stop_loss_threshold=None, learning_rate=0.01, delta=0.01, spsa_samples=128,
         spsa_iters=1, is_debug=False):
  """Tensorflow 2.0 implementation of SPSA.

  This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666 (Uesato et al. 2018).
  SPSA is a gradient-free optimization method, which is useful when the model is non-differentiable,
  or more generally, the gradients do not point in useful directions.
  :param model_fn: A callable that takes an input tensor and returns the model logits.
  :param x: Input tensor.
  :param y: Tensor with true labels. If targeted is true, then provide the target label.
  :param eps: The size of the maximum perturbation, measured in the L-infinity norm.
  :param nb_iter: The number of optimization steps.
  :param clip_min: If specified, the minimum input value.
  :param clip_max: If specified, the maximum input value.
  :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the default,
            will try to make the label incorrect. Targeted will instead try to move in the direction
            of being more like y.
  :param early_stop_loss_threshold: A float or None. If specified, the attack will end as soon as
                                    the loss is below `early_stop_loss_threshold`.
  :param learning_rate: Learning rate of ADAM optimizer.
  :param delta: Perturbation size used for SPSA approximation.
  :param spsa_samples:  Number of inputs to evaluate at a single time. The true batch size
                        (the number of evaluated inputs for each update) is `spsa_samples *
                        spsa_iters`
  :param spsa_iters:  Number of model evaluations before performing an update, where each evaluation
                      is on `spsa_samples` different inputs.
  :param is_debug: If True, print the adversarial loss after each update.
  """
  if x.get_shape().as_list()[0] != 1:
    raise ValueError("For SPSA, input tensor x must have batch_size of 1.")

  optimizer = SPSAAdam(lr=learning_rate, delta=delta, num_samples=spsa_samples,
                       num_iters=spsa_iters)

  def loss_fn(x, label):
    """
    Margin logit loss, with correct sign for targeted vs untargeted loss.
    """
    logits = model_fn(x)
    loss_multiplier = 1 if targeted else -1
    return loss_multiplier * margin_logit_loss(logits, label, nb_classes=logits.get_shape()[-1])

  adv_x = projected_optimization(loss_fn, x, y, eps, nb_iter, optimizer, clip_min, clip_max,
                                 early_stop_loss_threshold, is_debug=is_debug)

  return adv_x


class SPSAAdam(tf.optimizers.Adam):
  """Optimizer for gradient-free attacks in https://arxiv.org/abs/1802.05666.

  Gradients estimates are computed using Simultaneous Perturbation Stochastic Approximation (SPSA),
  combined with the ADAM update rule (https://arxiv.org/abs/1412.6980).
  """

  def __init__(self, lr=0.01, delta=0.01, num_samples=128, num_iters=1,
               compare_to_analytic_grad=False):
    super(SPSAAdam, self).__init__(lr=lr)
    assert num_samples % 2 == 0, "number of samples must be even"
    self._delta = delta
    self._num_samples = num_samples // 2  # Since we mirror +/- delta later
    self._num_iters = num_iters
    self._compare_to_analytic_grad = compare_to_analytic_grad

  def _get_delta(self, x, delta):
    x_shape = x.get_shape().as_list()
    delta_x = delta * tf.sign(
      tf.random.uniform([self._num_samples] + x_shape[1:], minval=-1., maxval=1., dtype=tf_dtype)
    )
    return delta_x

  def _compute_gradients(self, loss_fn, x):
    """Compute a new value of `x` to minimize `loss_fn` using SPSA.

    Args:
        loss_fn:  a callable that takes `x`, a batch of images, and returns a batch of loss values.
                  `x` will be optimized to minimize `loss_fn(x)`.
        x:  A list of Tensors, the values to be updated. This is analogous to the `var_list` argument
            in standard TF Optimizer.

    Returns:
        new_x: A list of Tensors, the same length as `x`, which are updated
        new_optim_state:  A dict, with the same structure as `optim_state`, which have been updated.
    """

    # Assumes `x` is a list, containing a [1, H, W, C] image.If static batch dimension is None,
    # tf.reshape to batch size 1 so that static shape can be inferred.
    assert len(x) == 1
    static_x_shape = x[0].get_shape().as_list()
    if static_x_shape[0] is None:
      x[0] = tf.reshape(x[0], [1] + static_x_shape[1:])
    assert x[0].get_shape().as_list()[0] == 1
    x = x[0]
    x_shape = x.get_shape().as_list()

    def body(i, grad_array):
      delta = self._delta
      delta_x = self._get_delta(x, delta)
      delta_x = tf.concat([delta_x, -delta_x], axis=0)
      loss_vals = tf.reshape(
        loss_fn(x + delta_x),
        [2 * self._num_samples] + [1] * (len(x_shape) - 1))
      avg_grad = tf.reduce_mean(loss_vals * delta_x, axis=0) / delta
      avg_grad = tf.expand_dims(avg_grad, axis=0)
      new_grad_array = grad_array.write(i, avg_grad)
      return i + 1, new_grad_array

    def cond(i, _):
      return i < self._num_iters

    _, all_grads = tf.while_loop(
      cond,
      body,
      loop_vars=[
        0, tf.TensorArray(size=self._num_iters, dtype=tf_dtype)
      ],
      back_prop=False,
      parallel_iterations=1)
    avg_grad = tf.reduce_sum(all_grads.stack(), axis=0)
    return [avg_grad]

  def _apply_gradients(self, grads, x, optim_state):
    """Given a gradient, make one optimization step.

    :param grads: list of tensors, same length as `x`, containing the corresponding gradients
    :param x: list of tensors to update
    :param optim_state: dict

    Returns:
      new_x: list of tensors, updated version of `x`
      new_optim_state: dict, updated version of `optim_state`
    """

    new_x = [None] * len(x)
    new_optim_state = {
      "t": optim_state["t"] + 1.,
      "m": [None] * len(x),
      "u": [None] * len(x)
    }
    t = new_optim_state["t"]
    for i in range(len(x)):
      g = grads[i]
      m_old = optim_state["m"][i]
      u_old = optim_state["u"][i]
      new_optim_state["m"][i] = (self.beta_1 * m_old + (1. - self.beta_1) * g)
      new_optim_state["u"][i] = (self.beta_2 * u_old + (1. - self.beta_2) * g * g)
      m_hat = new_optim_state["m"][i] / (1. - tf.pow(self.beta_1, t))
      u_hat = new_optim_state["u"][i] / (1. - tf.pow(self.beta_2, t))
      new_x[i] = (x[i] - self.lr * m_hat / (tf.sqrt(u_hat) + self.epsilon))
    return new_x, new_optim_state

  def init_state(self, x):
    """Initialize t, m, and u"""
    optim_state = {
      "t": 0.,
      "m": [tf.zeros_like(v) for v in x],
      "u": [tf.zeros_like(v) for v in x]
    }
    return optim_state

  def minimize(self, loss_fn, x, optim_state):
    """Analogous to tf.Optimizer.minimize

    :param loss_fn: tf Tensor, representing the loss to minimize
    :param x: list of Tensor, analogous to tf.Optimizer's var_list
    :param optim_state: A possibly nested dict, containing any optimizer state.

    Returns:
      new_x: list of Tensor, updated version of `x`
      new_optim_state: dict, updated version of `optim_state`
    """
    grads = self._compute_gradients(loss_fn, x)
    return self._apply_gradients(grads, x, optim_state)


def margin_logit_loss(model_logits, label, nb_classes=10):
  """Computes difference between logit for `label` and next highest logit.

  The loss is high when `label` is unlikely (targeted by default). This follows the same interface
  as `loss_fn` for projected_optimization, i.e. it returns a batch of loss values.
  """

  if 'int' in str(label.dtype):
    logit_mask = tf.one_hot(label, depth=nb_classes, axis=-1)
  else:
    logit_mask = label
  if 'int' in str(logit_mask.dtype):
    logit_mask = tf.cast(logit_mask, dtype=tf.float32)
  try:
    label_logits = tf.reduce_sum(logit_mask * model_logits, axis=-1)
  except TypeError:
    raise TypeError(
      "Could not take row-wise dot product between logit mask, of dtype " + str(logit_mask.dtype)
      + " and model_logits, of dtype " + str(model_logits.dtype)
    )
  logits_with_target_label_neg_inf = model_logits - logit_mask * 99999
  highest_nonlabel_logits = tf.reduce_max(logits_with_target_label_neg_inf, axis=-1)
  loss = highest_nonlabel_logits - label_logits
  return loss


def _project_perturbation(perturbation, epsilon, input_image, clip_min=None, clip_max=None):
  """
  Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into hypercube such
  that the resulting adversarial example is between clip_min and clip_max, if applicable.
  """

  if clip_min is None or clip_max is None:
    raise NotImplementedError("_project_perturbation currently has clipping hard-coded in.")

  # Ensure inputs are in the correct range
  with tf.control_dependencies([
    tf.debugging.assert_less_equal(input_image, tf.cast(clip_max, input_image.dtype)),
    tf.debugging.assert_greater_equal(input_image, tf.cast(clip_min, input_image.dtype))
  ]):
    clipped_perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
    new_image = tf.clip_by_value(input_image + clipped_perturbation, clip_min, clip_max)
    return new_image - input_image


def projected_optimization(loss_fn, input_image, label, epsilon, num_steps, optimizer,
                           clip_min=None, clip_max=None, early_stop_loss_threshold=None,
                           project_perturbation=_project_perturbation,
                           is_debug=False):
  """
  Generic projected optimization, generalized to work with approximate gradients. Used for e.g.
  the SPSA attack.

  Args:
    :param loss_fn: A callable which takes `input_image` and `label` as
                    arguments, and returns a batch of loss values.
    :param input_image: Tensor, a batch of images
    :param label: Tensor, a batch of labels
    :param epsilon: float, the L-infinity norm of the maximum allowable
                    perturbation
    :param num_steps: int, the number of steps of gradient descent
    :param optimizer: A `SPSAAdam` object
    :param clip_min: float, minimum pixel value
    :param clip_max: float, maximum pixel value
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
    adversarial version of `input_image`, with L-infinity difference less than epsilon, which tries
    to minimize loss_fn.

  Note that this function is not intended as an Attack by itself. Rather, it is designed as a helper
  function which you can use to write your own attack methods. The method uses a tf.while_loop to
  optimize a loss function in a single sess.run() call.
  """
  assert num_steps is not None
  if is_debug:
    with tf.device("/cpu:0"):
      tf.print("Starting PGD attack with epsilon: %s" % epsilon)

  init_perturbation = tf.random.uniform(tf.shape(input_image),
                                        minval=tf.cast(-epsilon, input_image.dtype),
                                        maxval=tf.cast(epsilon, input_image.dtype),
                                        dtype=input_image.dtype)
  init_perturbation = project_perturbation(init_perturbation, epsilon, input_image,
                                           clip_min=clip_min, clip_max=clip_max)
  init_optim_state = optimizer.init_state([init_perturbation])

  def loop_body(i, perturbation, flat_optim_state):
    """Update perturbation to input image."""
    optim_state = tf.nest.pack_sequence_as(structure=init_optim_state,
                                           flat_sequence=flat_optim_state)

    def wrapped_loss_fn(x):
      return loss_fn(input_image + x, label)

    new_perturbation_list, new_optim_state = optimizer.minimize(wrapped_loss_fn, [perturbation],
                                                                optim_state)
    projected_perturbation = project_perturbation(new_perturbation_list[0], epsilon, input_image,
                                                  clip_min=clip_min, clip_max=clip_max)

    # Be careful with this bool. A value of 0. is a valid threshold but evaluates to False, so we
    # must explicitly check whether the value is None.
    early_stop = early_stop_loss_threshold is not None
    compute_loss = is_debug or early_stop
    # Don't waste time building the loss graph if we're not going to use it
    if compute_loss:
      # NOTE: this step is not actually redundant with the optimizer step.
      # SPSA calculates the loss at randomly perturbed points but doesn't calculate the loss at the current point.
      loss = tf.reduce_mean(wrapped_loss_fn(projected_perturbation), axis=0)

      if is_debug:
        with tf.device("/cpu:0"):
          tf.print(loss, "Total batch loss")

      if early_stop:
        i = tf.cond(tf.less(loss, early_stop_loss_threshold), lambda: float(num_steps), lambda: i)

    return i + 1, projected_perturbation, tf.nest.flatten(new_optim_state)

  def cond(i, *_):
    return tf.less(i, num_steps)

  flat_init_optim_state = tf.nest.flatten(init_optim_state)
  _, final_perturbation, _ = tf.while_loop(
    cond,
    loop_body,
    loop_vars=(tf.constant(0.), init_perturbation, flat_init_optim_state),
    parallel_iterations=1,
    back_prop=False,
    maximum_iterations=num_steps
  )

  if project_perturbation is _project_perturbation:
    # TODO: this assert looks totally wrong.
    # Not bothering to fix it now because it's only an assert.
    # 1) Multiplying by 1.1 gives a huge margin of error. This should probably take the difference
    #    and allow a tolerance of 1e-6 or something like that.
    # 2) I think it should probably check the *absolute value* of final_perturbation
    perturbation_max = epsilon * 1.1
    check_diff = tf.debugging.assert_less_equal(
      final_perturbation,
      tf.cast(perturbation_max, final_perturbation.dtype),
      message="final_perturbation must change no pixel by more than %s" % perturbation_max
    )
  else:
    # TODO: let caller pass in a check_diff function as well as
    # project_perturbation
    check_diff = tf.no_op()

  if clip_min is None or clip_max is None:
    raise NotImplementedError("This function only supports clipping for now")
  check_range = [
    tf.debugging.assert_less_equal(input_image, tf.cast(clip_max, input_image.dtype)),
    tf.debugging.assert_greater_equal(input_image, tf.cast(clip_min, input_image.dtype))
  ]

  with tf.control_dependencies([check_diff] + check_range):
    adversarial_image = input_image + final_perturbation
  return tf.stop_gradient(adversarial_image)
