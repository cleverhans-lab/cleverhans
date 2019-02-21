# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import numpy as np

from cleverhans.attacks.carlini_wagner_l2 import CWL2 as CarliniWagnerL2 # pylint: disable=unused-import
from cleverhans.attacks.deep_fool import deepfool_batch, deepfool_attack # pylint: disable=unused-import
from cleverhans.attacks.elastic_net_method import EAD as ElasticNetMethod # pylint: disable=unused-import
from cleverhans.attacks.lbfgs import LBFGS_impl as LBFGS_attack # pylint: disable=unused-import
from cleverhans.attacks.saliency_map_method import jsma_symbolic # pylint: disable=unused-import
from cleverhans.attacks.spsa import TensorOptimizer, TensorGradientDescent, TensorAdam # pylint: disable=unused-import
from cleverhans.attacks.spsa import SPSAAdam, margin_logit_loss, _apply_black_border # pylint: disable=unused-import
from cleverhans.attacks.spsa import _apply_transformation, spm, parallel_apply_transformations # pylint: disable=unused-import
from cleverhans.attacks.virtual_adversarial_method import vatm # pylint: disable=unused-import
from cleverhans.utils_tf import jacobian_graph, jacobian_augmentation # pylint: disable=unused-import
from cleverhans import utils

np_dtype = np.dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.tf")

warnings.warn("attacks_tf is deprecated and will be removed on 2019-07-18"
              " or after. Code should import functions from their new locations directly.")


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
  warnings.warn(
      "This function is dead code and will be removed on or after 2019-07-18")

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
  warnings.warn(
      "This function is dead code and will be removed on or after 2019-07-18")

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
  warnings.warn(
      "This function is dead code and will be removed on or after 2019-07-18")

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


class UnrolledOptimizer(TensorOptimizer):
  def __init__(self, *args, **kwargs):
    warnings.warn("UnrolledOptimizer has been renamed to TensorOptimizer."
                  " The old name may be removed on or after 2019-04-25.")
    super(UnrolledOptimizer, self).__init__(*args, **kwargs)


class UnrolledGradientDescent(TensorGradientDescent):
  def __init__(self, *args, **kwargs):
    warnings.warn("UnrolledGradientDescent has been renamed to "
                  "TensorGradientDescent."
                  " The old name may be removed on or after 2019-04-25.")
    super(UnrolledGradientDescent, self).__init__(*args, **kwargs)


class UnrolledAdam(TensorAdam):
  def __init__(self, *args, **kwargs):
    warnings.warn("UnrolledAdam has been renamed to TensorAdam."
                  " The old name may be removed on or after 2019-04-25.")
    super(UnrolledAdam, self).__init__(*args, **kwargs)


def pgd_attack(*args, **kwargs):
  warnings.warn("cleverhans.attacks_tf.pgd_attack has been renamed to "
                "cleverhans.attacks.projected_optimization. "
                "Please switch to the new name. The current name will "
                "become unsupport on or after 2019-04-24.")
  from cleverhans.attacks import projected_optimization
  return projected_optimization(*args, **kwargs)
