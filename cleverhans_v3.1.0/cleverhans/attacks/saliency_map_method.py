"""The SalienceMapMethod attack
"""
# pylint: disable=missing-docstring
import warnings

import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_sum, reduce_max, reduce_any

tf_dtype = tf.as_dtype('float32')


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


def jsma_symbolic(x, y_target, model, theta, gamma, clip_min, clip_max):
  """
  TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
  for details about the algorithm design choices).

  :param x: the input placeholder
  :param y_target: the target tensor
  :param model: a cleverhans.model.Model object.
  :param theta: delta for each feature adjustment
  :param gamma: a float between 0 - 1 indicating the maximum distortion
      percentage
  :param clip_min: minimum value for components of the example returned
  :param clip_max: maximum value for components of the example returned
  :return: a tensor for the adversarial example
  """

  nb_classes = int(y_target.shape[-1].value)
  nb_features = int(np.product(x.shape[1:]).value)

  if x.dtype == tf.float32 and y_target.dtype == tf.int64:
    y_target = tf.cast(y_target, tf.int32)

  if x.dtype == tf.float32 and y_target.dtype == tf.float64:
    warnings.warn("Downcasting labels---this should be harmless unless"
                  " they are smoothed")
    y_target = tf.cast(y_target, tf.float32)

  max_iters = np.floor(nb_features * gamma / 2)
  increase = bool(theta > 0)

  tmp = np.ones((nb_features, nb_features), int)
  np.fill_diagonal(tmp, 0)
  zero_diagonal = tf.constant(tmp, tf_dtype)

  # Compute our initial search domain. We optimize the initial search domain
  # by removing all features that are already at their maximum values (if
  # increasing input features---otherwise, at their minimum value).
  if increase:
    search_domain = tf.reshape(
        tf.cast(x < clip_max, tf_dtype), [-1, nb_features])
  else:
    search_domain = tf.reshape(
        tf.cast(x > clip_min, tf_dtype), [-1, nb_features])

  # Loop variables
  # x_in: the tensor that holds the latest adversarial outputs that are in
  #       progress.
  # y_in: the tensor for target labels
  # domain_in: the tensor that holds the latest search domain
  # cond_in: the boolean tensor to show if more iteration is needed for
  #          generating adversarial samples
  def condition(x_in, y_in, domain_in, i_in, cond_in):
    # Repeat the loop until we have achieved misclassification or
    # reaches the maximum iterations
    return tf.logical_and(tf.less(i_in, max_iters), cond_in)

  # Same loop variables as above
  def body(x_in, y_in, domain_in, i_in, cond_in):
    # Create graph for model logits and predictions
    logits = model.get_logits(x_in)
    preds = tf.nn.softmax(logits)
    preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

    # create the Jacobian graph
    list_derivatives = []
    for class_ind in xrange(nb_classes):
      derivatives = tf.gradients(logits[:, class_ind], x_in)
      list_derivatives.append(derivatives[0])
    grads = tf.reshape(
        tf.stack(list_derivatives), shape=[nb_classes, -1, nb_features])

    # Compute the Jacobian components
    # To help with the computation later, reshape the target_class
    # and other_class to [nb_classes, -1, 1].
    # The last dimention is added to allow broadcasting later.
    target_class = tf.reshape(
        tf.transpose(y_in, perm=[1, 0]), shape=[nb_classes, -1, 1])
    other_classes = tf.cast(tf.not_equal(target_class, 1), tf_dtype)

    grads_target = reduce_sum(grads * target_class, axis=0)
    grads_other = reduce_sum(grads * other_classes, axis=0)

    # Remove the already-used input features from the search space
    # Subtract 2 times the maximum value from those value so that
    # they won't be picked later
    increase_coef = (4 * int(increase) - 2) \
        * tf.cast(tf.equal(domain_in, 0), tf_dtype)

    target_tmp = grads_target
    target_tmp -= increase_coef \
        * reduce_max(tf.abs(grads_target), axis=1, keepdims=True)
    target_sum = tf.reshape(target_tmp, shape=[-1, nb_features, 1]) \
        + tf.reshape(target_tmp, shape=[-1, 1, nb_features])

    other_tmp = grads_other
    other_tmp += increase_coef \
        * reduce_max(tf.abs(grads_other), axis=1, keepdims=True)
    other_sum = tf.reshape(other_tmp, shape=[-1, nb_features, 1]) \
        + tf.reshape(other_tmp, shape=[-1, 1, nb_features])

    # Create a mask to only keep features that match conditions
    if increase:
      scores_mask = ((target_sum > 0) & (other_sum < 0))
    else:
      scores_mask = ((target_sum < 0) & (other_sum > 0))

    # Create a 2D numpy array of scores for each pair of candidate features
    scores = tf.cast(scores_mask, tf_dtype) \
        * (-target_sum * other_sum) * zero_diagonal

    # Extract the best two pixels
    best = tf.argmax(
        tf.reshape(scores, shape=[-1, nb_features * nb_features]), axis=1)

    p1 = tf.mod(best, nb_features)
    p2 = tf.floordiv(best, nb_features)
    p1_one_hot = tf.one_hot(p1, depth=nb_features)
    p2_one_hot = tf.one_hot(p2, depth=nb_features)

    # Check if more modification is needed for each sample
    mod_not_done = tf.equal(reduce_sum(y_in * preds_onehot, axis=1), 0)
    cond = mod_not_done & (reduce_sum(domain_in, axis=1) >= 2)

    # Update the search domain
    cond_float = tf.reshape(tf.cast(cond, tf_dtype), shape=[-1, 1])
    to_mod = (p1_one_hot + p2_one_hot) * cond_float

    domain_out = domain_in - to_mod

    # Apply the modification to the images
    to_mod_reshape = tf.reshape(
        to_mod, shape=([-1] + x_in.shape[1:].as_list()))
    if increase:
      x_out = tf.minimum(clip_max, x_in + to_mod_reshape * theta)
    else:
      x_out = tf.maximum(clip_min, x_in - to_mod_reshape * theta)

    # Increase the iterator, and check if all misclassifications are done
    i_out = tf.add(i_in, 1)
    cond_out = reduce_any(cond)

    return x_out, y_in, domain_out, i_out, cond_out

  # Run loop to do JSMA
  x_adv, _, _, _, _ = tf.while_loop(
      condition,
      body, [x, y_target, search_domain, 0, True],
      parallel_iterations=1)

  return x_adv
