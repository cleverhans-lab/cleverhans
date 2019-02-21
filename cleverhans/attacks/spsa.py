"""The SPSA attack
"""
# pylint: disable=missing-docstring
import warnings

import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_mean, reduce_sum, reduce_max
from cleverhans.model import Model
from cleverhans import utils_tf

tf_dtype = tf.as_dtype('float32')


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
      """
      Margin logit loss, with correct sign for targeted vs untargeted loss.
      """
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


class TensorOptimizer(object):
  """Optimizer for Tensors rather than tf.Variables.

  TensorOptimizers implement optimizers where the values being optimized
  are ordinary Tensors, rather than Variables. TF Variables can have strange
  behaviors when being assigned multiple times within a single sess.run()
  call, particularly in Distributed TF, so this avoids thinking about those
  issues. These are helper classes for the `projected_optimization`
  method. Apart from not using Variables, they follow an interface very
  similar to tf.Optimizer.
  """

  def _compute_gradients(self, loss_fn, x, unused_optim_state):
    """Compute a new value of `x` to minimize `loss_fn`.

    Args:
        loss_fn: a callable that takes `x`, a batch of images, and returns
            a batch of loss values. `x` will be optimized to minimize
            `loss_fn(x)`.
        x: A list of Tensors, the values to be updated. This is analogous
            to the `var_list` argument in standard TF Optimizer.
        unused_optim_state: A (possibly nested) dict, containing any state
            info needed for the optimizer.

    Returns:
        new_x: A list of Tensors, the same length as `x`, which are updated
        new_optim_state: A dict, with the same structure as `optim_state`,
            which have been updated.
    """

    # Assumes `x` is a list,
    # and contains a tensor representing a batch of images
    assert len(x) == 1 and isinstance(x, list), \
        'x should be a list and contain only one image tensor'
    x = x[0]
    loss = reduce_mean(loss_fn(x), axis=0)
    return tf.gradients(loss, x)

  def _apply_gradients(self, grads, x, optim_state):
    """
    Given a gradient, make one optimization step.

    :param grads: list of tensors, same length as `x`, containing the corresponding gradients
    :param x: list of tensors to update
    :param optim_state: dict

    Returns:
      new_x: list of tensors, updated version of `x`
      new_optim_state: dict, updated version of `optim_state`
    """
    raise NotImplementedError(
        "_apply_gradients should be defined in each subclass")

  def minimize(self, loss_fn, x, optim_state):
    """
    Analogous to tf.Optimizer.minimize

    :param loss_fn: tf Tensor, representing the loss to minimize
    :param x: list of Tensor, analogous to tf.Optimizer's var_list
    :param optim_state: A possibly nested dict, containing any optimizer state.

    Returns:
      new_x: list of Tensor, updated version of `x`
      new_optim_state: dict, updated version of `optim_state`
    """
    grads = self._compute_gradients(loss_fn, x, optim_state)
    return self._apply_gradients(grads, x, optim_state)

  def init_state(self, x):
    """Returns the initial state of the optimizer.

    Args:
        x: A list of Tensors, which will be optimized.

    Returns:
        A dictionary, representing the initial state of the optimizer.
    """
    raise NotImplementedError(
        "init_state should be defined in each subclass")


class TensorGradientDescent(TensorOptimizer):
  """Vanilla Gradient Descent TensorOptimizer."""

  def __init__(self, lr):
    self._lr = lr

  def init_state(self, x):
    return {}

  def _apply_gradients(self, grads, x, optim_state):
    new_x = [None] * len(x)
    for i in xrange(len(x)):
      new_x[i] = x[i] - self._lr * grads[i]
    return new_x, optim_state


class TensorAdam(TensorOptimizer):
  """The Adam optimizer defined in https://arxiv.org/abs/1412.6980."""

  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-9):
    self._lr = lr
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

  def init_state(self, x):
    """
    Initialize t, m, and u
    """
    optim_state = {}
    optim_state["t"] = 0.
    optim_state["m"] = [tf.zeros_like(v) for v in x]
    optim_state["u"] = [tf.zeros_like(v) for v in x]
    return optim_state

  def _apply_gradients(self, grads, x, optim_state):
    """Refer to parent class documentation."""
    new_x = [None] * len(x)
    new_optim_state = {
        "t": optim_state["t"] + 1.,
        "m": [None] * len(x),
        "u": [None] * len(x)
    }
    t = new_optim_state["t"]
    for i in xrange(len(x)):
      g = grads[i]
      m_old = optim_state["m"][i]
      u_old = optim_state["u"][i]
      new_optim_state["m"][i] = (
          self._beta1 * m_old + (1. - self._beta1) * g)
      new_optim_state["u"][i] = (
          self._beta2 * u_old + (1. - self._beta2) * g * g)
      m_hat = new_optim_state["m"][i] / (1. - tf.pow(self._beta1, t))
      u_hat = new_optim_state["u"][i] / (1. - tf.pow(self._beta2, t))
      new_x[i] = (
          x[i] - self._lr * m_hat / (tf.sqrt(u_hat) + self._epsilon))
    return new_x, new_optim_state


class SPSAAdam(TensorAdam):
  """Optimizer for gradient-free attacks in https://arxiv.org/abs/1802.05666.

  Gradients estimates are computed using Simultaneous Perturbation Stochastic
  Approximation (SPSA), combined with the ADAM update rule.
  """

  def __init__(self,
               lr=0.01,
               delta=0.01,
               num_samples=128,
               num_iters=1,
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
        tf.random_uniform(
            [self._num_samples] + x_shape[1:],
            minval=-1.,
            maxval=1.,
            dtype=tf_dtype))
    return delta_x

  def _compute_gradients(self, loss_fn, x, unused_optim_state):
    """Compute gradient estimates using SPSA."""
    # Assumes `x` is a list, containing a [1, H, W, C] image
    # If static batch dimension is None, tf.reshape to batch size 1
    # so that static shape can be inferred
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
      avg_grad = reduce_mean(loss_vals * delta_x, axis=0) / delta
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
    avg_grad = reduce_sum(all_grads.stack(), axis=0)
    return [avg_grad]


def margin_logit_loss(model_logits, label, nb_classes=10, num_classes=None):
  """Computes difference between logit for `label` and next highest logit.

  The loss is high when `label` is unlikely (targeted by default).
  This follows the same interface as `loss_fn` for TensorOptimizer and
  projected_optimization, i.e. it returns a batch of loss values.
  """
  if num_classes is not None:
    warnings.warn("`num_classes` is depreciated. Switch to `nb_classes`."
                  " `num_classes` may be removed on or after 2019-04-23.")
    nb_classes = num_classes
    del num_classes
  if 'int' in str(label.dtype):
    logit_mask = tf.one_hot(label, depth=nb_classes, axis=-1)
  else:
    logit_mask = label
  if 'int' in str(logit_mask.dtype):
    logit_mask = tf.to_float(logit_mask)
  try:
    label_logits = reduce_sum(logit_mask * model_logits, axis=-1)
  except TypeError:
    raise TypeError("Could not take row-wise dot product between "
                    "logit mask, of dtype " + str(logit_mask.dtype)
                    + " and model_logits, of dtype "
                    + str(model_logits.dtype))
  logits_with_target_label_neg_inf = model_logits - logit_mask * 99999
  highest_nonlabel_logits = reduce_max(
      logits_with_target_label_neg_inf, axis=-1)
  loss = highest_nonlabel_logits - label_logits
  return loss


def _apply_black_border(x, border_size):
  orig_height = x.get_shape().as_list()[1]
  orig_width = x.get_shape().as_list()[2]
  x = tf.image.resize_images(x, (orig_width - 2*border_size,
                                 orig_height - 2*border_size))

  return tf.pad(x, [[0, 0],
                    [border_size, border_size],
                    [border_size, border_size],
                    [0, 0]], 'CONSTANT')


def _apply_transformation(inputs):
  x, trans = inputs[0], inputs[1]
  dx, dy, angle = trans[0], trans[1], trans[2]
  height = x.get_shape().as_list()[1]
  width = x.get_shape().as_list()[2]

  # Pad the image to prevent two-step rotation / translation from truncating
  # corners
  max_dist_from_center = np.sqrt(height**2+width**2) / 2
  min_edge_from_center = float(np.min([height, width])) / 2
  padding = np.ceil(max_dist_from_center -
                    min_edge_from_center).astype(np.int32)
  x = tf.pad(x, [[0, 0],
                 [padding, padding],
                 [padding, padding],
                 [0, 0]],
             'CONSTANT')

  # Apply rotation
  angle *= np.pi / 180
  x = tf.contrib.image.rotate(x, angle, interpolation='BILINEAR')

  # Apply translation
  dx_in_px = -dx * height
  dy_in_px = -dy * width
  translation = tf.convert_to_tensor([dx_in_px, dy_in_px])

  try:
    x = tf.contrib.image.translate(x, translation, interpolation='BILINEAR')
  except AttributeError as e:
    print("WARNING: SpatialAttack requires tf 1.6 or higher")
    raise e
  x = tf.contrib.image.translate(x, translation, interpolation='BILINEAR')
  return tf.image.resize_image_with_crop_or_pad(x, height, width)


def spm(x, model, y=None, n_samples=None, dx_min=-0.1,
        dx_max=0.1, n_dxs=5, dy_min=-0.1, dy_max=0.1, n_dys=5,
        angle_min=-30, angle_max=30, n_angles=31, black_border_size=0):
  """
  TensorFlow implementation of the Spatial Transformation Method.
  :return: a tensor for the adversarial example
  """
  if y is None:
    preds = model.get_probs(x)
    # Using model predictions as ground truth to avoid label leaking
    preds_max = reduce_max(preds, 1, keepdims=True)
    y = tf.to_float(tf.equal(preds, preds_max))
    y = tf.stop_gradient(y)
    del preds
  y = y / reduce_sum(y, 1, keepdims=True)

  # Define the range of transformations
  dxs = np.linspace(dx_min, dx_max, n_dxs)
  dys = np.linspace(dy_min, dy_max, n_dys)
  angles = np.linspace(angle_min, angle_max, n_angles)

  if n_samples is None:
    import itertools
    transforms = list(itertools.product(*[dxs, dys, angles]))
  else:
    sampled_dxs = np.random.choice(dxs, n_samples)
    sampled_dys = np.random.choice(dys, n_samples)
    sampled_angles = np.random.choice(angles, n_samples)
    transforms = zip(sampled_dxs, sampled_dys, sampled_angles)
  transformed_ims = parallel_apply_transformations(
      x, transforms, black_border_size)

  def _compute_xent(x):
    preds = model.get_logits(x)
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y, logits=preds)

  all_xents = tf.map_fn(
      _compute_xent,
      transformed_ims,
      parallel_iterations=1)  # Must be 1 to avoid keras race conditions

  # Return the adv_x with worst accuracy

  # all_xents is n_total_samples x batch_size (SB)
  all_xents = tf.stack(all_xents)  # SB

  # We want the worst case sample, with the largest xent_loss
  worst_sample_idx = tf.argmax(all_xents, axis=0)  # B

  batch_size = tf.shape(x)[0]
  keys = tf.stack([
      tf.range(batch_size, dtype=tf.int32),
      tf.cast(worst_sample_idx, tf.int32)
  ], axis=1)
  transformed_ims_bshwc = tf.einsum('sbhwc->bshwc', transformed_ims)
  after_lookup = tf.gather_nd(transformed_ims_bshwc, keys)  # BHWC
  return after_lookup


def parallel_apply_transformations(x, transforms, black_border_size=0):
  """
  Apply image transformations in parallel.
  :param transforms: TODO
  :param black_border_size: int, size of black border to apply
  Returns:
    Transformed images
  """
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  x = _apply_black_border(x, black_border_size)

  num_transforms = transforms.get_shape().as_list()[0]
  im_shape = x.get_shape().as_list()[1:]

  # Pass a copy of x and a transformation to each iteration of the map_fn
  # callable
  tiled_x = tf.reshape(
      tf.tile(x, [num_transforms, 1, 1, 1]),
      [num_transforms, -1] + im_shape)
  elems = [tiled_x, transforms]
  transformed_ims = tf.map_fn(
      _apply_transformation,
      elems,
      dtype=tf.float32,
      parallel_iterations=1,  # Must be 1 to avoid keras race conditions
  )
  return transformed_ims


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
        i = tf.cond(tf.less(loss, early_stop_loss_threshold),
                    lambda: float(num_steps), lambda: i)

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
