from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings

from . import utils_tf
from . import utils

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def fgsm(x, predictions, eps=0.3, clip_min=None, clip_max=None):
    return fgm(x, predictions, y=None, eps=eps, ord=np.inf, clip_min=clip_min,
               clip_max=clip_max)


def fgm(x, preds, y=None, eps=0.3, ord=np.inf, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor
    :param y: (optional) A placeholder for the model labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        signed_grad = tf.sign(grad)
    elif ord == 1:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.reduce_sum(tf.abs(grad),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True)
    elif ord == 2:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                                   reduction_indices=reduc_ind,
                                                   keep_dims=True))
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def vatm(model, x, logits, eps, num_iterations=1, xi=1e-6,
         clip_min=None, clip_max=None, scope=None):
    """
    Tensorflow implementation of the perturbation method used for virtual
    adversarial training: https://arxiv.org/abs/1507.00677
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor
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
        d = tf.random_normal(tf.shape(x))
        for i in range(num_iterations):
            d = xi * utils_tf.l2_batch_normalize(d)
            logits_d = model(x + d)
            kl = utils_tf.kl_with_logits(logits, logits_d)
            Hd = tf.gradients(kl, d)[0]
            d = tf.stop_gradient(Hd)
        d = eps * utils_tf.l2_batch_normalize(d)
        adv_x = tf.stop_gradient(x + d)
        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        return adv_x


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
    # Compute the size of the input (the number of features)
    nf = len(grads_target)

    # Remove the already-used input features from the search space
    invalid = list(set(range(nf)) - search_domain)
    increase_coef = (2 * int(increase) - 1)
    grads_target[invalid] = - increase_coef * np.max(np.abs(grads_target))
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


def jacobian(sess, x, grads, target, X, nb_features, nb_classes):
    """
    TensorFlow implementation of the foward derivative / Jacobian
    :param x: the input placeholder
    :param grads: the list of TF gradients returned by jacobian_graph()
    :param target: the target misclassification class
    :param X: numpy array with sample input
    :param nb_features: the number of features in the input
    :return: matrix of forward derivatives flattened into vectors
    """
    # Prepare feeding dictionary for all gradient computations
    feed_dict = {x: X}

    # Initialize a numpy array to hold the Jacobian component values
    jacobian_val = np.zeros((nb_classes, nb_features), dtype=np.float32)

    # Compute the gradients for all classes
    for class_ind, grad in enumerate(grads):
        run_grad = sess.run(grad, feed_dict)
        jacobian_val[class_ind] = np.reshape(run_grad, (1, nb_features))

    # Sum over all classes different from the target class to prepare for
    # saliency map computation in the next step of the attack
    other_classes = utils.other_classes(nb_classes, target)
    grad_others = np.sum(jacobian_val[other_classes, :], axis=0)

    return jacobian_val[target], grad_others


def jacobian_graph(predictions, x, nb_classes):
    """
    Create the Jacobian graph to be ran later in a TF session
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param x: the input placeholder
    :param nb_classes: the number of classes the model has
    :return:
    """
    # This function will return a list of TF gradients
    list_derivatives = []

    # Define the TF graph elements to compute our derivatives for each class
    for class_ind in xrange(nb_classes):
        derivatives, = tf.gradients(predictions[:, class_ind], x)
        list_derivatives.append(derivatives)

    return list_derivatives


def jsma(sess, x, predictions, grads, sample, target, theta, gamma, clip_min,
         clip_max):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param grads: symbolic gradients
    :param sample: numpy array with sample input
    :param target: target class for sample input
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :return: an adversarial sample
    """

    # Copy the source sample and define the maximum number of features
    # (i.e. the maximum number of iterations) that we may perturb
    adv_x = copy.copy(sample)
    # count the number of features. For MNIST, 1x28x28 = 784; for
    # CIFAR, 3x32x32 = 3072; etc.
    nb_features = np.product(adv_x.shape[1:])
    # reshape sample for sake of standardization
    original_shape = adv_x.shape
    adv_x = np.reshape(adv_x, (1, nb_features))
    # compute maximum number of iterations
    max_iters = np.floor(nb_features * gamma / 2)

    # Find number of classes based on grads
    nb_classes = len(grads)

    increase = bool(theta > 0)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] < clip_max])
    else:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] > clip_min])

    # Initialize the loop variables
    iteration = 0
    adv_x_original_shape = np.reshape(adv_x, original_shape)
    current = utils_tf.model_argmax(sess, x, predictions, adv_x_original_shape)

    # Repeat this main loop until we have achieved misclassification
    while (current != target and iteration < max_iters and
           len(search_domain) > 1):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_original_shape,
                                              nb_features, nb_classes)

        # Compute the saliency map for each of our target classes
        # and return the two best candidate features for perturbation
        i, j, search_domain = saliency_map(
            grads_target, grads_others, search_domain, increase)

        # Apply the perturbation to the two input features selected previously
        adv_x = apply_perturbations(
            i, j, adv_x, increase, theta, clip_min, clip_max)

        # Update our current prediction by querying the model
        current = utils_tf.model_argmax(sess, x, predictions,
                                        adv_x_original_shape)

        # Update loop variables
        iteration = iteration + 1

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


def jsma_batch(sess, x, pred, grads, X, theta, gamma, clip_min, clip_max,
               nb_classes, targets=None):
    """
    Applies the JSMA to a batch of inputs
    :param sess: TF session
    :param x: the input placeholder
    :param pred: the model's symbolic output
    :param grads: symbolic gradients
    :param X: numpy array with sample inputs
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :param nb_classes: number of model output classes
    :param targets: target class for sample input
    :return: adversarial examples
    """
    X_adv = np.zeros(X.shape)

    for ind, val in enumerate(X):
        val = np.expand_dims(val, axis=0)
        if targets is None:
            # No targets provided, randomly choose from other classes
            from .utils_tf import model_argmax
            gt = model_argmax(sess, x, pred, val)

            # Randomly choose from the incorrect classes for each sample
            from .utils import random_targets
            target = random_targets(gt, nb_classes)[0]
        else:
            target = targets[ind]

        X_adv[ind], _, _ = jsma(sess, x, pred, grads, val, np.argmax(target),
                                theta, gamma, clip_min, clip_max)

    return np.asarray(X_adv, dtype=np.float32)


def jacobian_augmentation(sess, x, X_sub_prev, Y_sub, grads, lmbda,
                          keras_phase=None):
    """
    Augment an adversary's substitute training set using the Jacobian
    of a substitute model to generate new synthetic inputs.
    See https://arxiv.org/abs/1602.02697 for more details.
    See tutorials/mnist_blackbox.py for example use case
    :param sess: TF session in which the substitute model is defined
    :param x: input TF placeholder for the substitute model
    :param X_sub_prev: substitute training data available to the adversary
                       at the previous iteration
    :param Y_sub: substitute training labels available to the adversary
                  at the previous iteration
    :param grads: Jacobian symbolic graph for the substitute
                  (should be generated using attacks_tf.jacobian_graph)
    :param keras_phase: (deprecated) if not None, holds keras learning_phase
    :return: augmented substitute data (will need to be labeled by oracle)
    """
    assert len(x.get_shape()) == len(np.shape(X_sub_prev))
    assert len(grads) >= np.max(Y_sub) + 1
    assert len(X_sub_prev) == len(Y_sub)

    if keras_phase is not None:
        warnings.warn("keras_phase argument is deprecated and will be removed"
                      " on 2017-09-28. Instead, use K.set_learning_phase(0) at"
                      " the start of your script and serve with tensorflow.")

    # Prepare input_shape (outside loop) for feeding dictionary below
    input_shape = list(x.get_shape())
    input_shape[0] = 1

    # Create new numpy array for adversary training data
    # with twice as many components on the first dimension.
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, input in enumerate(X_sub_prev):
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Prepare feeding dictionary
        feed_dict = {x: np.reshape(input, input_shape)}

        # Compute sign matrix
        grad_val = sess.run([tf.sign(grad)], feed_dict=feed_dict)[0]

        # Create new synthetic point in adversary substitute training set
        X_sub[2*ind] = X_sub[ind] + lmbda * grad_val

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub
