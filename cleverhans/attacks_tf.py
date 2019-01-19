from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import warnings

import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks.carlini_wagner_l2 import CWL2 as CarliniWagnerL2
from cleverhans.attacks.deep_fool import deepfool_batch, deepfool_attack
from cleverhans.attacks.elastic_net_method import EAD as ElasticNetMethod
from cleverhans.attacks.saliency_map_method import jsma_symbolic
from cleverhans.attacks.virtual_adversarial_method import vatm
from cleverhans.utils_tf import jacobian_graph, jacobian_augmentation
from cleverhans.compat import reduce_max
from cleverhans.compat import reduce_mean, reduce_sum
from cleverhans.compat import reduce_any
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans import utils_tf
from cleverhans import utils

_logger = utils.create_logger("cleverhans.attacks.tf")

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

import warnings

warnings.warn("attacks_tf is deprecated and will be removed on 2019-07-18 or after. Code should import functions from their new locations directly.")


def fgsm(x, predictions, eps=0.3, clip_min=None, clip_max=None):
  warnings.warn("This function is deprecated and will be removed on or after "
                "2019-04-09. Switch to cleverhans.attacks.FastGradientMethod.")
  return fgm(
      x,
      predictions,
      y=None,
      eps=eps,
      ord=np.inf,
      clip_min=clip_min,
      clip_max=clip_max)


def fgm(x, preds, *args, **kwargs):
  if preds.op.type == 'Softmax':
    logits, = preds.op.inputs
  else:
    raise TypeError("Unclear how to get logits")
  warnings.warn("This function is deprecated. Switch to passing *logits* to"
                " cleverhans.attacks.fgm")
  from cleverhans.attacks import fgm as logits_fgm
  return logits_fgm(x, logits, *args, **kwargs)


def apply_perturbations(i, j, X, increase, theta, clip_min, clip_max):
  """
  TensorFlow implementation for apply perturbations to input features based
  on salency maps
  :param i: index of first selected feature
  :param j: index of second selected feature
  :param X: a matrix containing our input features for our sample
  :param increase: boolean; true if we are increasing pixels, false otherwise
  :param theta: delta for each feature adjustment
  :param clip_min: mininum value for a feature in our sample
  :param clip_max: maximum value for a feature in our sample
  : return: a perturbed input feature matrix for a target class
  """
  warnings.warn("This function is dead code and will be removed on or after 2019-07-18")

  # perturb our input sample
  if increase:
    X[0, i] = np.minimum(clip_max, X[0, i] + theta)
    X[0, j] = np.minimum(clip_max, X[0, j] + theta)
  else:
    X[0, i] = np.maximum(clip_min, X[0, i] - theta)
    X[0, j] = np.maximum(clip_min, X[0, j] - theta)

  return X


def saliency_map(grads_target, grads_other, search_domain, increase):
  """
  TensorFlow implementation for computing saliency maps
  :param grads_target: a matrix containing forward derivatives for the
                       target class
  :param grads_other: a matrix where every element is the sum of forward
                      derivatives over all non-target classes at that index
  :param search_domain: the set of input indices that we are considering
  :param increase: boolean; true if we are increasing pixels, false otherwise
  :return: (i, j, search_domain) the two input indices selected and the
           updated search domain
  """
  warnings.warn("This function is dead code and will be removed on or after 2019-07-18")

  # Compute the size of the input (the number of features)
  nf = len(grads_target)

  # Remove the already-used input features from the search space
  invalid = list(set(range(nf)) - search_domain)
  increase_coef = (2 * int(increase) - 1)
  grads_target[invalid] = -increase_coef * np.max(np.abs(grads_target))
  grads_other[invalid] = increase_coef * np.max(np.abs(grads_other))

  # Create a 2D numpy array of the sum of grads_target and grads_other
  target_sum = grads_target.reshape((1, nf)) + grads_target.reshape((nf, 1))
  other_sum = grads_other.reshape((1, nf)) + grads_other.reshape((nf, 1))

  # Create a mask to only keep features that match saliency map conditions
  if increase:
    scores_mask = ((target_sum > 0) & (other_sum < 0))
  else:
    scores_mask = ((target_sum < 0) & (other_sum > 0))

  # Create a 2D numpy array of the scores for each pair of candidate features
  scores = scores_mask * (-target_sum * other_sum)

  # A pixel can only be selected (and changed) once
  np.fill_diagonal(scores, 0)

  # Extract the best two pixels
  best = np.argmax(scores)
  p1, p2 = best % nf, best // nf

  # Remove used pixels from our search domain
  search_domain.discard(p1)
  search_domain.discard(p2)

  return p1, p2, search_domain


def jacobian(sess, x, grads, target, X, nb_features, nb_classes, feed=None):
  """
  TensorFlow implementation of the foward derivative / Jacobian
  :param x: the input placeholder
  :param grads: the list of TF gradients returned by jacobian_graph()
  :param target: the target misclassification class
  :param X: numpy array with sample input
  :param nb_features: the number of features in the input
  :return: matrix of forward derivatives flattened into vectors
  """
  warnings.warn("This function is dead code and will be removed on or after 2019-07-18")

  # Prepare feeding dictionary for all gradient computations
  feed_dict = {x: X}
  if feed is not None:
    feed_dict.update(feed)

  # Initialize a numpy array to hold the Jacobian component values
  jacobian_val = np.zeros((nb_classes, nb_features), dtype=np_dtype)

  # Compute the gradients for all classes
  for class_ind, grad in enumerate(grads):
    run_grad = sess.run(grad, feed_dict)
    jacobian_val[class_ind] = np.reshape(run_grad, (1, nb_features))

  # Sum over all classes different from the target class to prepare for
  # saliency map computation in the next step of the attack
  other_classes = utils.other_classes(nb_classes, target)
  grad_others = np.sum(jacobian_val[other_classes, :], axis=0)

  return jacobian_val[target], grad_others







class LBFGS_attack(object):
  def __init__(self, sess, x, model_preds, targeted_label,
               binary_search_steps, max_iterations, initial_const, clip_min,
               clip_max, nb_classes, batch_size):
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.

    :param sess: a TF session.
    :param x: A tensor with the inputs.
    :param model_preds: A tensor with model's predictions.
    :param targeted_label: A tensor with the target labels.
    :param binary_search_steps: The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the purturbation
                                and cross-entropy loss of classification.
    :param max_iterations: The maximum number of iterations.
    :param initial_const: The initial tradeoff-constant to use to tune the
                          relative importance of size of the purturbation
                          and cross-entropy loss of the classification.
    :param clip_min: Minimum input component value
    :param clip_max: Maximum input component value
    :param num_labels: The number of classes in the model's output.
    :param batch_size: Number of attacks to run simultaneously.

    """
    warnings.warn("This class is deprecated and will be removed on or after "
                  "2019-04-10. Switch to cleverhans.attacks.LBFGS_impl. "
                  "Note that it uses *logits* not *probabilities*.")
    self.sess = sess
    self.x = x
    self.model_preds = model_preds
    self.targeted_label = targeted_label
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.batch_size = batch_size

    self.repeat = self.binary_search_steps >= 10
    self.shape = tuple([self.batch_size] +
                       list(self.x.get_shape().as_list()[1:]))
    self.ori_img = tf.Variable(
        np.zeros(self.shape), dtype=tf_dtype, name='ori_img')
    self.const = tf.Variable(
        np.zeros(self.batch_size), dtype=tf_dtype, name='const')

    assert self.model_preds.op.type == 'Softmax'
    logits, = self.model_preds.op.inputs
    self.score = softmax_cross_entropy_with_logits(labels=self.targeted_label,
                                                   logits=logits)
    self.l2dist = reduce_sum(tf.square(self.x - self.ori_img))
    # small self.const will result small adversarial perturbation
    self.loss = reduce_sum(self.score * self.const) + self.l2dist
    self.grad, = tf.gradients(self.loss, self.x)

  def attack(self, x_val, targets):
    """
    Perform the attack on the given instance for the given targets.
    """

    def lbfgs_objective(adv_x, self, targets, oimgs, CONST):
      # returns the function value and the gradient for fmin_l_bfgs_b
      loss = self.sess.run(
          self.loss,
          feed_dict={
              self.x: adv_x.reshape(oimgs.shape),
              self.targeted_label: targets,
              self.ori_img: oimgs,
              self.const: CONST
          })
      grad = self.sess.run(
          self.grad,
          feed_dict={
              self.x: adv_x.reshape(oimgs.shape),
              self.targeted_label: targets,
              self.ori_img: oimgs,
              self.const: CONST
          })
      return loss, grad.flatten().astype(float)

    # begin the main part for the attack
    from scipy.optimize import fmin_l_bfgs_b
    oimgs = np.clip(x_val, self.clip_min, self.clip_max)
    CONST = np.ones(self.batch_size) * self.initial_const

    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(self.batch_size)
    upper_bound = np.ones(self.batch_size) * 1e10

    # set the box constraints for the optimization function
    clip_min = self.clip_min * np.ones(oimgs.shape[:])
    clip_max = self.clip_max * np.ones(oimgs.shape[:])
    clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))

    # placeholders for the best l2 and instance attack found so far
    o_bestl2 = [1e10] * self.batch_size
    o_bestattack = np.copy(oimgs)

    for outer_step in range(self.binary_search_steps):
      _logger.debug("  Binary search step %s of %s",
                    outer_step, self.binary_search_steps)

      # The last iteration (if we run many steps) repeat the search once.
      if self.repeat and outer_step == self.binary_search_steps - 1:
        CONST = upper_bound

      # optimization function
      adv_x, _, __ = fmin_l_bfgs_b(
          lbfgs_objective,
          oimgs.flatten().astype(float),
          args=(self, targets, oimgs, CONST),
          bounds=clip_bound,
          maxiter=self.max_iterations,
          iprint=0)

      adv_x = adv_x.reshape(oimgs.shape)
      assert np.amax(adv_x) <= self.clip_max and \
          np.amin(adv_x) >= self.clip_min, \
          'fmin_l_bfgs_b returns are invalid'

      # adjust the best result (i.e., the adversarial example with the
      # smallest perturbation in terms of L_2 norm) found so far
      preds = np.atleast_1d(
          utils_tf.model_argmax(self.sess, self.x, self.model_preds,
                                adv_x))
      _logger.debug("predicted labels are %s", preds)

      l2s = np.zeros(self.batch_size)
      for i in range(self.batch_size):
        l2s[i] = np.sum(np.square(adv_x[i] - oimgs[i]))

      for e, (l2, pred, ii) in enumerate(zip(l2s, preds, adv_x)):
        if l2 < o_bestl2[e] and pred == np.argmax(targets[e]):
          o_bestl2[e] = l2
          o_bestattack[e] = ii

      # adjust the constant as needed
      for e in range(self.batch_size):
        if preds[e] == np.argmax(targets[e]):
          # success, divide const by two
          upper_bound[e] = min(upper_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
        else:
          # failure, either multiply by 10 if no solution found yet
          #          or do binary search with the known upper bound
          lower_bound[e] = max(lower_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
          else:
            CONST[e] *= 10

      _logger.debug("  Successfully generated adversarial examples " +
                    "on {} of {} instances.".format(
                        sum(upper_bound < 1e9), self.batch_size))
      o_bestl2 = np.array(o_bestl2)
      mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
      _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

    # return the best solution found
    o_bestl2 = np.array(o_bestl2)
    return o_bestattack


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
    raise NotImplementedError(
        "_apply_gradients should be defined in each subclass")

  def minimize(self, loss_fn, x, optim_state):
    grads = self._compute_gradients(loss_fn, x, optim_state)
    return self._apply_gradients(grads, x, optim_state)

  def init_optim_state(self, x):
    """Returns the initial state of the optimizer.

    Args:
        x: A list of Tensors, which will be optimized.

    Returns:
        A dictionary, representing the initial state of the optimizer.
    """
    raise NotImplementedError(
        "init_optim_state should be defined in each subclass")

class UnrolledOptimizer(TensorOptimizer):
  def __init__(self, *args, **kwargs):
    warnings.warn("UnrolledOptimizer has been renamed to TensorOptimizer."
                  " The old name may be removed on or after 2019-04-25.")
    super(UnrolledOptimizer, self).__init__(*args, **kwargs)


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

class UnrolledGradientDescent(TensorGradientDescent):
  def __init__(self, *args, **kwargs):
    warnings.warn("UnrolledGradientDescent has been renamed to "
                  "TensorGradientDescent."
                  " The old name may be removed on or after 2019-04-25.")
    super(UnrolledGradientDescent, self).__init__(*args, **kwargs)


class TensorAdam(TensorOptimizer):
  """The Adam optimizer defined in https://arxiv.org/abs/1412.6980."""

  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-9):
    self._lr = lr
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

  def init_state(self, x):
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


class UnrolledAdam(TensorAdam):
  def __init__(self, *args, **kwargs):
    warnings.warn("UnrolledAdam has been renamed to TensorAdam."
                  " The old name may be removed on or after 2019-04-25.")
    super(UnrolledAdam, self).__init__(*args, **kwargs)


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




def pgd_attack(*args, **kwargs):
  warnings.warn("cleverhans.attacks_tf.pgd_attack has been renamed to "
                "cleverhans.attacks.projected_optimization. "
                "Please switch to the new name. The current name will "
                "become unsupport on or after 2019-04-24.")
  from cleverhans.attacks import projected_optimization
  return projected_optimization(*args, **kwargs)



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
