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

_logger = utils.create_logger("cleverhans.attacks.tf")

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')


def ZERO():
    return np.asarray(0., dtype=np_dtype)


def fgsm(x, predictions, eps=0.3, clip_min=None, clip_max=None):
    return fgm(x, predictions, y=None, eps=eps, ord=np.inf, clip_min=clip_min,
               clip_max=clip_max)


def fgm(x, preds, y=None, eps=0.3, ord=np.inf,
        clip_min=None, clip_max=None,
        targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(xrange(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
    elif ord == 2:
        red_ind = list(xrange(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

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
        for i in range(num_iterations):
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
         clip_max, feed=None):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output (the attack expects the
                  probabilities, i.e., the output of the softmax, but will
                  also work with logits typically)
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
    current = utils_tf.model_argmax(sess, x, predictions, adv_x_original_shape,
                                    feed=feed)

    _logger.debug("Starting JSMA attack up to {} iterations".format(max_iters))
    # Repeat this main loop until we have achieved misclassification
    while (current != target and iteration < max_iters and
           len(search_domain) > 1):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_original_shape,
                                              nb_features, nb_classes,
                                              feed=feed)

        if iteration % ((max_iters + 1) // 5) == 0 and iteration > 0:
            _logger.debug("Iteration {} of {}".format(iteration,
                                                      int(max_iters)))
        # Compute the saliency map for each of our target classes
        # and return the two best candidate features for perturbation
        i, j, search_domain = saliency_map(
            grads_target, grads_others, search_domain, increase)

        # Apply the perturbation to the two input features selected previously
        adv_x = apply_perturbations(
            i, j, adv_x, increase, theta, clip_min, clip_max)

        # Update our current prediction by querying the model
        current = utils_tf.model_argmax(sess, x, predictions,
                                        adv_x_original_shape, feed=feed)

        # Update loop variables
        iteration = iteration + 1

    if current == target:
        _logger.info("Attack succeeded using {} iterations".format(iteration))
    else:
        _logger.info(("Failed to find adversarial example " +
                      "after {} iterations").format(iteration))

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


def jsma_batch(sess, x, pred, grads, X, theta, gamma, clip_min, clip_max,
               nb_classes, y_target=None, feed=None, **kwargs):
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
    :param y_target: target class for sample input
    :return: adversarial examples
    """

    warnings.warn("jsma_batch is deprecated and will be removed on "
                  "2018-06-10. Use jsma_symbolic instead.")

    X_adv = np.zeros(X.shape)

    for ind, val in enumerate(X):
        val = np.expand_dims(val, axis=0)
        if y_target is None:
            # No y_target provided, randomly choose from other classes
            from .utils_tf import model_argmax
            gt = model_argmax(sess, x, pred, val, feed=feed)

            # Randomly choose from the incorrect classes for each sample
            from .utils import random_targets
            target = random_targets(gt, nb_classes)[0]
        else:
            target = y_target[ind]

        X_adv[ind], _, _ = jsma(sess, x, pred, grads, val, np.argmax(target),
                                theta, gamma, clip_min, clip_max, feed=feed)

    return np.asarray(X_adv, dtype=np_dtype)


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
                            tf.cast(x < clip_max, tf_dtype),
                            [-1, nb_features])
    else:
        search_domain = tf.reshape(
                            tf.cast(x > clip_min, tf_dtype),
                            [-1, nb_features])

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

        preds = model.get_probs(x_in)
        preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

        # create the Jacobian graph
        list_derivatives = []
        for class_ind in xrange(nb_classes):
            derivatives = tf.gradients(preds[:, class_ind], x_in)
            list_derivatives.append(derivatives[0])
        grads = tf.reshape(tf.stack(list_derivatives),
                           shape=[nb_classes, -1, nb_features])

        # Compute the Jacobian components
        # To help with the computation later, reshape the target_class
        # and other_class to [nb_classes, -1, 1].
        # The last dimention is added to allow broadcasting later.
        target_class = tf.reshape(tf.transpose(y_in, perm=[1, 0]),
                                  shape=[nb_classes, -1, 1])
        other_classes = tf.cast(tf.not_equal(target_class, 1), tf_dtype)

        grads_target = tf.reduce_sum(grads * target_class, axis=0)
        grads_other = tf.reduce_sum(grads * other_classes, axis=0)

        # Remove the already-used input features from the search space
        # Subtract 2 times the maximum value from those value so that
        # they won't be picked later
        increase_coef = (4 * int(increase) - 2) \
            * tf.cast(tf.equal(domain_in, 0), tf_dtype)

        target_tmp = grads_target
        target_tmp -= increase_coef \
            * tf.reduce_max(tf.abs(grads_target), axis=1, keep_dims=True)
        target_sum = tf.reshape(target_tmp, shape=[-1, nb_features, 1]) \
            + tf.reshape(target_tmp, shape=[-1, 1, nb_features])

        other_tmp = grads_other
        other_tmp += increase_coef \
            * tf.reduce_max(tf.abs(grads_other), axis=1, keep_dims=True)
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
                    tf.reshape(scores, shape=[-1, nb_features * nb_features]),
                    axis=1)

        p1 = tf.mod(best, nb_features)
        p2 = tf.floordiv(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        p2_one_hot = tf.one_hot(p2, depth=nb_features)

        # Check if more modification is needed for each sample
        mod_not_done = tf.equal(tf.reduce_sum(y_in * preds_onehot, axis=1), 0)
        cond = mod_not_done & (tf.reduce_sum(domain_in, axis=1) >= 2)

        # Update the search domain
        cond_float = tf.reshape(tf.cast(cond, tf_dtype), shape=[-1, 1])
        to_mod = (p1_one_hot + p2_one_hot) * cond_float

        domain_out = domain_in - to_mod

        # Apply the modification to the images
        to_mod_reshape = tf.reshape(to_mod,
                                    shape=([-1] + x_in.shape[1:].as_list()))
        if increase:
            x_out = tf.minimum(clip_max, x_in + to_mod_reshape * theta)
        else:
            x_out = tf.maximum(clip_min, x_in - to_mod_reshape * theta)

        # Increase the iterator, and check if all misclassifications are done
        i_out = tf.add(i_in, 1)
        cond_out = tf.reduce_any(cond)

        return x_out, y_in, domain_out, i_out, cond_out

    # Run loop to do JSMA
    x_adv, _, _, _, _ = tf.while_loop(condition, body,
                                      [x, y_target, search_domain, 0, True],
                                      parallel_iterations=1)

    return x_adv


def jacobian_augmentation(sess, x, X_sub_prev, Y_sub, grads, lmbda,
                          feed=None):
    """
    Augment an adversary's substitute training set using the Jacobian
    of a substitute model to generate new synthetic inputs.
    See https://arxiv.org/abs/1602.02697 for more details.
    See cleverhans_tutorials/mnist_blackbox.py for example use case
    :param sess: TF session in which the substitute model is defined
    :param x: input TF placeholder for the substitute model
    :param X_sub_prev: substitute training data available to the adversary
                       at the previous iteration
    :param Y_sub: substitute training labels available to the adversary
                  at the previous iteration
    :param grads: Jacobian symbolic graph for the substitute
                  (should be generated using attacks_tf.jacobian_graph)
    :return: augmented substitute data (will need to be labeled by oracle)
    """
    assert len(x.get_shape()) == len(np.shape(X_sub_prev))
    assert len(grads) >= np.max(Y_sub) + 1
    assert len(X_sub_prev) == len(Y_sub)

    # Prepare input_shape (outside loop) for feeding dictionary below
    input_shape = list(x.get_shape())
    input_shape[0] = 1

    # Create new numpy array for adversary training data
    # with twice as many components on the first dimension.
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, prev_input in enumerate(X_sub_prev):
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Prepare feeding dictionary
        feed_dict = {x: np.reshape(prev_input, input_shape)}
        if feed is not None:
            feed_dict.update(feed)

        # Compute sign matrix
        grad_val = sess.run([tf.sign(grad)], feed_dict=feed_dict)[0]

        # Create new synthetic point in adversary substitute training set
        X_sub[X_sub_prev.shape[0] + ind] = X_sub[ind] + lmbda * grad_val

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub


class CarliniWagnerL2(object):

    def __init__(self, sess, model, batch_size, confidence,
                 targeted, learning_rate,
                 binary_search_steps, max_iterations,
                 abort_early, initial_const,
                 clip_min, clip_max, num_labels, shape):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial
                         examples produced. If set to False, they will be
                         misclassified in any wrong class. If set to True,
                         they will be misclassified in a chosen target class.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model

        self.repeat = binary_search_steps >= 10

        self.shape = shape = tuple([batch_size] + list(shape))

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype,
                                name='timg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)),
                                dtype=tf_dtype, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf_dtype,
                                 name='const')

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf_dtype, shape,
                                          name='assign_timg')
        self.assign_tlab = tf.placeholder(tf_dtype, (batch_size, num_labels),
                                          name='assign_tlab')
        self.assign_const = tf.placeholder(tf_dtype, [batch_size],
                                           name='assign_const')

        # the resulting instance, tanh'd to keep bounded from clip_min
        # to clip_max
        self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
        self.newimg = self.newimg * (clip_max - clip_min) + clip_min

        # prediction BEFORE-SOFTMAX of the model
        self.output = model.get_logits(self.newimg)

        # distance to the input data
        self.other = (tf.tanh(self.timg) + 1) / \
            2 * (clip_max - clip_min) + clip_min
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.other),
                                    list(range(1, len(shape))))

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max(
            (1 - self.tlab) * self.output - self.tlab * 10000,
            1)

        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * loss1)
        self.loss = self.loss1 + self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
        for i in range(0, len(imgs), self.batch_size):
            _logger.debug(("Running CWL2 attack on instance " +
                           "{} of {}").format(i, len(imgs)))
            r.extend(self.attack_batch(imgs[i:i + self.batch_size],
                                       targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of instance and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        oimgs = np.clip(imgs, self.clip_min, self.clip_max)

        # re-scale instances to be within range [0, 1]
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        # now convert to [-1, 1]
        imgs = (imgs * 2) - 1
        # convert to tanh-space
        imgs = np.arctanh(imgs * .999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(oimgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            _logger.debug("  Binary search step {} of {}".
                          format(outer_step, self.BINARY_SEARCH_STEPS))

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg = self.sess.run([self.train,
                                                         self.loss,
                                                         self.l2dist,
                                                         self.output,
                                                         self.newimg])

                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                                   "l2={:.3g} f={:.3g}")
                                  .format(iteration, self.MAX_ITERATIONS,
                                          l, np.mean(l2s), np.mean(scores)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                   iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                   bestscore[e] != -1:
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
                          "on {} of {} instances.".
                          format(sum(upper_bound < 1e9), batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack


class ElasticNetMethod(object):

    def __init__(self, sess, model, fista, beta,
                 decision_rule, batch_size, confidence,
                 targeted, learning_rate,
                 binary_search_steps, max_iterations,
                 abort_early, initial_const,
                 clip_min, clip_max, num_labels, shape):
        """
        EAD Attack

        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param fista: FISTA or ISTA. FISTA has better convergence properties
                      but performs an additional query per iteration
        :param beta: Trades off L2 distortion with L1 distortion: higher
                     produces examples with lower L1 distortion, at the
                     cost of higher L2 (and typically Linf) distortion
        :param decision_rule: EN or L1. Select final adversarial example from
                              all successful examples based on the least
                              elastic-net or L1 distortion criterion.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial
                         examples produced. If set to False, they will be
                         misclassified in any wrong class. If set to True,
                         they will be misclassified in a chosen target class.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the perturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early abort when the total
                            loss starts to increase (greatly speeds up attack,
                            but hurts performance, particularly on ImageNet)
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the perturbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model
        self.decision_rule = decision_rule

        self.fista = fista
        self.beta = beta
        self.beta_t = tf.cast(self.beta, tf_dtype)

        self.repeat = binary_search_steps >= 10

        self.shape = shape = tuple([batch_size] + list(shape))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype,
                                name='timg')
        self.newimg = tf.Variable(np.zeros(shape), dtype=tf_dtype,
                                  name='newimg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)),
                                dtype=tf_dtype, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf_dtype,
                                 name='const')

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf_dtype, shape,
                                          name='assign_timg')
        self.assign_newimg = tf.placeholder(tf_dtype, shape,
                                            name='assign_newimg')
        self.assign_tlab = tf.placeholder(tf_dtype, (batch_size, num_labels),
                                          name='assign_tlab')
        self.assign_const = tf.placeholder(tf_dtype, [batch_size],
                                           name='assign_const')

        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_t = tf.cast(self.global_step, tf_dtype)

        if self.fista:
            self.slack = tf.Variable(np.zeros(shape), dtype=tf_dtype,
                                     name='slack')
            self.assign_slack = tf.placeholder(tf_dtype, shape,
                                               name='assign_slack')
            var = self.slack
        else:
            var = self.newimg

        """Fast Iterative Shrinkage Thresholding"""
        """--------------------------------"""
        self.zt = tf.divide(self.global_step_t,
                            self.global_step_t + tf.cast(3, tf_dtype))

        cond1 = tf.cast(tf.greater(tf.subtract(var, self.timg),
                                   self.beta_t), tf_dtype)
        cond2 = tf.cast(tf.less_equal(tf.abs(tf.subtract(var,
                                                         self.timg)),
                                      self.beta_t), tf_dtype)
        cond3 = tf.cast(tf.less(tf.subtract(var, self.timg),
                                tf.negative(self.beta_t)), tf_dtype)

        upper = tf.minimum(tf.subtract(var, self.beta_t),
                           tf.cast(self.clip_max, tf_dtype))
        lower = tf.maximum(tf.add(var, self.beta_t),
                           tf.cast(self.clip_min, tf_dtype))

        self.assign_newimg = tf.multiply(cond1, upper)
        self.assign_newimg += tf.multiply(cond2, self.timg)
        self.assign_newimg += tf.multiply(cond3, lower)
        self.setter = tf.assign(self.newimg, self.assign_newimg)
        if self.fista:
            self.assign_slack = self.assign_newimg
            self.assign_slack += tf.multiply(self.zt,
                                             self.assign_newimg - self.newimg)
            self.setter_y = tf.assign(self.slack, self.assign_slack)
        """--------------------------------"""

        # prediction BEFORE-SOFTMAX of the model
        self.output = model.get_logits(self.newimg)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-self.timg),
                                    list(range(1, len(shape))))
        self.l1dist = tf.reduce_sum(tf.abs(self.newimg-self.timg),
                                    list(range(1, len(shape))))
        self.elasticdist = self.l2dist + tf.multiply(self.l1dist,
                                                     self.beta_t)
        if self.decision_rule == 'EN':
            self.crit = self.elasticdist
            self.crit_p = 'Elastic'
        else:
            self.crit = self.l1dist
            self.crit_p = 'L1'

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max((1 - self.tlab) * self.output -
                              (self.tlab * 10000), 1)

        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss21 = tf.reduce_sum(self.l1dist)
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * loss1)

        if self.fista:
            self.output_y = model.get_logits(self.slack)
            self.l2dist_y = tf.reduce_sum(tf.square(self.slack-self.timg),
                                          list(range(1, len(shape))))
            real_y = tf.reduce_sum((self.tlab) * self.output_y, 1)
            other_y = tf.reduce_max((1 - self.tlab) * self.output_y -
                                    (self.tlab * 10000), 1)
            if self.TARGETED:
                loss1_y = tf.maximum(ZERO(),
                                     other_y - real_y + self.CONFIDENCE)
            else:
                loss1_y = tf.maximum(ZERO(),
                                     real_y - other_y + self.CONFIDENCE)

            self.loss2_y = tf.reduce_sum(self.l2dist_y)
            self.loss1_y = tf.reduce_sum(self.const * loss1_y)

            self.loss_opt = self.loss1_y+self.loss2_y
        else:
            self.loss_opt = self.loss1+self.loss2
        self.loss = self.loss1+self.loss2+tf.multiply(self.beta_t, self.loss21)

        self.learning_rate = tf.train.polynomial_decay(self.LEARNING_RATE,
                                                       self.global_step,
                                                       self.MAX_ITERATIONS,
                                                       0, power=0.5)

        # Setup the optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train = optimizer.minimize(self.loss_opt,
                                        var_list=[var],
                                        global_step=self.global_step)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        var_list = [self.global_step] + [self.newimg] + new_vars
        if self.fista:
            var_list += [self.slack]
        self.init = tf.variables_initializer(var_list=var_list)

    def attack(self, imgs, targets):
        """
        Perform the EAD attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        batch_size = self.batch_size
        r = []
        for i in range(0, len(imgs) // batch_size):
            _logger.debug(("Running EAD attack on instance " +
                           "{} of {}").format(i * batch_size, len(imgs)))
            r.extend(self.attack_batch(imgs[i*batch_size:(i+1)*batch_size],
                                       targets[i*batch_size:(i+1)*batch_size]))
        if len(imgs) % batch_size != 0:
            last_elements = len(imgs) - (len(imgs) % batch_size)
            _logger.debug(("Running EAD attack on instance " +
                           "{} of {}").format(last_elements, len(imgs)))
            temp_imgs = np.zeros((batch_size,) + imgs.shape[2:])
            temp_targets = np.zeros((batch_size,) + targets.shape[2:])
            temp_imgs[:(len(imgs) % batch_size)] = imgs[last_elements:]
            temp_targets[:(len(imgs) % batch_size)] = targets[last_elements:]
            temp_data = self.attack_batch(temp_imgs, temp_targets)
            r.extend(temp_data[:(len(imgs) % batch_size)],
                     targets[last_elements:])
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of instance and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        imgs = np.clip(imgs, self.clip_min, self.clip_max)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # placeholders for the best en, score, and instance attack found so far
        o_bestdst = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(imgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset the optimizer's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestdst = [1e10] * batch_size
            bestscore = [-1] * batch_size
            _logger.debug("  Binary search step {} of {}".
                          format(outer_step, self.BINARY_SEARCH_STEPS))

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            self.sess.run(self.setter, {self.assign_newimg: batch})
            if self.fista:
                self.sess.run(self.setter_y, {self.assign_slack: batch})
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                self.sess.run([self.train])
                if self.fista:
                    _, _, l, l2s, l1s, crit = self.sess.run([self.setter,
                                                             self.setter_y,
                                                             self.loss,
                                                             self.l2dist,
                                                             self.l1dist,
                                                             self.crit])
                else:
                    _, l, l2s, l1s, crit = self.sess.run([self.setter,
                                                          self.loss,
                                                          self.l2dist,
                                                          self.l1dist,
                                                          self.crit])
                scores, nimg = self.sess.run([self.output, self.newimg])

                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                                   "l2={:.3g} l1={:.3g} f={:.3g}")
                                  .format(iteration, self.MAX_ITERATIONS,
                                          l, np.mean(l2s), np.mean(l1s),
                                          np.mean(scores)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                   iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l

                # adjust the best result found so far
                for e, (dst, sc, ii) in enumerate(zip(crit, scores, nimg)):
                    lab = np.argmax(batchlab[e])
                    if dst < bestdst[e] and compare(sc, lab):
                        bestdst[e] = dst
                        bestscore[e] = np.argmax(sc)
                    if dst < o_bestdst[e] and compare(sc, lab):
                        o_bestdst[e] = dst
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                   bestscore[e] != -1:
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
                          "on {} of {} instances.".
                          format(sum(upper_bound < 1e9), batch_size))
            o_bestdst = np.array(o_bestdst)
            mean = np.mean(np.sqrt(o_bestdst[o_bestdst < 1e9]))
            _logger.debug(self.crit_p + " Mean successful distortion: {:.4g}".
                          format(mean))

        # return the best solution found
        o_bestdst = np.array(o_bestdst)
        return o_bestattack


def deepfool_batch(sess, x, pred, logits, grads, X, nb_candidate, overshoot,
                   max_iter, clip_min, clip_max, nb_classes, feed=None):
    """
    Applies DeepFool to a batch of inputs
    :param sess: TF session
    :param x: The input placeholder
    :param pred: The model's sorted symbolic output of logits, only the top
                 nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                  from gradient_graph
    :param X: Numpy array with sample inputs
    :param nb_candidate: The number of classes to test against, i.e.,
                         deepfool only consider nb_candidate classes when
                         attacking(thus accelerate speed). The nb_candidate
                         classes are chosen according to the prediction
                         confidence during implementation.
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :param nb_classes: Number of model output classes
    :return: Adversarial examples
    """
    X_adv = deepfool_attack(sess, x, pred, logits, grads, X, nb_candidate,
                            overshoot, max_iter, clip_min, clip_max, feed=feed)

    return np.asarray(X_adv, dtype=np_dtype)


def deepfool_attack(sess, x, predictions, logits, grads, sample, nb_candidate,
                    overshoot, max_iter, clip_min, clip_max, feed=None):
    """
    TensorFlow implementation of DeepFool.
    Paper link: see https://arxiv.org/pdf/1511.04599.pdf
    :param sess: TF session
    :param x: The input placeholder
    :param predictions: The model's sorted symbolic output of logits, only the
                       top nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                 from gradient_graph
    :param sample: Numpy array with sample input
    :param nb_candidate: The number of classes to test against, i.e.,
                         deepfool only consider nb_candidate classes when
                         attacking(thus accelerate speed). The nb_candidate
                         classes are chosen according to the prediction
                         confidence during implementation.
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :return: Adversarial examples
    """
    import copy

    adv_x = copy.copy(sample)
    # Initialize the loop variables
    iteration = 0
    current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
    if current.shape == ():
        current = np.array([current])
    w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
    r_tot = np.zeros(sample.shape)
    original = current  # use original label as the reference

    _logger.debug("Starting DeepFool attack up to {} iterations".
                  format(max_iter))
    # Repeat this main loop until we have achieved misclassification
    while (np.any(current == original) and iteration < max_iter):

        if iteration % 5 == 0 and iteration > 0:
            _logger.info("Attack result at iteration {} is {}".format(
                iteration,
                current))
        gradients = sess.run(grads, feed_dict={x: adv_x})
        predictions_val = sess.run(predictions, feed_dict={x: adv_x})
        for idx in range(sample.shape[0]):
            pert = np.inf
            if current[idx] != original[idx]:
                continue
            for k in range(1, nb_candidate):
                w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                # adding value 0.00001 to prevent f_k = 0
                pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = pert*w/np.linalg.norm(w)
            r_tot[idx, ...] = r_tot[idx, ...] + r_i

        adv_x = np.clip(r_tot + sample, clip_min, clip_max)
        current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
        if current.shape == ():
            current = np.array([current])
        # Update loop variables
        iteration = iteration + 1

    # need more revision, including info like how many succeed
    _logger.info("Attack result at iteration {} is {}".format(iteration,
                 current))
    _logger.info("{} out of {}".format(sum(current != original),
                                       sample.shape[0]) +
                 " becomes adversarial examples at iteration {}".format(
                     iteration))
    # need to clip this image into the given range
    adv_x = np.clip((1+overshoot)*r_tot + sample, clip_min, clip_max)
    return adv_x


class LBFGS_attack(object):

    def __init__(self, sess, x, model_preds, targeted_label,
                 binary_search_steps, max_iterations, initial_const,
                 clip_min, clip_max, nb_classes, batch_size):
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
        self.shape = shape = tuple([self.batch_size] +
                                   list(self.x.get_shape().as_list()[1:]))
        self.ori_img = tf.Variable(np.zeros(self.shape), dtype=tf_dtype,
                                   name='ori_img')
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf_dtype,
                                 name='const')
        self.score = utils_tf.model_loss(self.targeted_label, self.model_preds,
                                         mean=False)
        self.l2dist = tf.reduce_sum(tf.square(self.x - self.ori_img))
        # small self.const will result small adversarial perturbation
        self.loss = tf.reduce_sum(self.score*self.const) + self.l2dist
        self.grad, = tf.gradients(self.loss, self.x)

    def attack(self, x_val, targets):
        """
        Perform the attack on the given instance for the given targets.
        """
        def lbfgs_objective(adv_x, self, targets, oimgs, CONST):
            # returns the function value and the gradient for fmin_l_bfgs_b
            loss = self.sess.run(self.loss, feed_dict={
                                 self.x: adv_x.reshape(oimgs.shape),
                                 self.targeted_label: targets,
                                 self.ori_img: oimgs,
                                 self.const: CONST})
            grad = self.sess.run(self.grad, feed_dict={
                                 self.x: adv_x.reshape(oimgs.shape),
                                 self.targeted_label: targets,
                                 self.ori_img: oimgs,
                                 self.const: CONST})
            return loss, grad.flatten().astype(float)

        # begin the main part for the attack
        from scipy.optimize import fmin_l_bfgs_b
        oimgs = np.clip(x_val, self.clip_min, self.clip_max)
        CONST = np.ones(self.batch_size) * self.initial_const

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(self.batch_size)
        upper_bound = np.ones(self.batch_size) * 1e10

        # set the box constraints for the optimization function
        clip_min = self.clip_min*np.ones(oimgs.shape[:])
        clip_max = self.clip_max*np.ones(oimgs.shape[:])
        clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))

        # placeholders for the best l2 and instance attack found so far
        o_bestl2 = [1e10] * self.batch_size
        o_bestattack = np.copy(oimgs)

        for outer_step in range(self.binary_search_steps):
            _logger.debug(("  Binary search step {} of {}")
                          .format(outer_step, self.binary_search_steps))

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.binary_search_steps - 1:
                CONST = upper_bound

            # optimization function
            adv_x, f, _ = fmin_l_bfgs_b(lbfgs_objective,
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
            preds = np.atleast_1d(utils_tf.model_argmax(self.sess, self.x,
                                                        self.model_preds,
                                                        adv_x))
            _logger.debug("predicted labels are {}".format(preds))

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
                          "on {} of {} instances.".
                          format(sum(upper_bound < 1e9), self.batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack


class UnrolledOptimizer(object):
    """Functional-stype optimizer which does not use TF Variables.

    UnrolledOptimizers implement optimizers where the values being optimized
    are ordinary Tensors, rather than Variables. TF Variables can have strange
    behaviors when being assigned multiple times within a single sess.run()
    call, particularly in Distributed TF, so this avoids thinking about those
    issues. In cleverhans, these are helper classes for the `pgd_attack`
    method.
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
        loss = tf.reduce_mean(loss_fn(x), axis=0)
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


class UnrolledGradientDescent(UnrolledOptimizer):
    """Vanilla Gradient Descent UnrolledOptimizer."""

    def __init__(self, lr):
        self._lr = lr

    def _apply_gradients(self, grads, x, optim_state):
        new_x = [None] * len(x)
        for i in xrange(len(x)):
            new_x[i] = x[i] - self._lr * grads[i]
        return new_x, optim_state


class UnrolledAdam(UnrolledOptimizer):
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


class SPSAAdam(UnrolledAdam):
    """Optimizer for gradient-free attacks in https://arxiv.org/abs/1802.05666.

    Gradients estimates are computed using Simultaneous Perturbation Stochastic
    Approximation (SPSA), combined with the ADAM update rule.
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
        delta_x = delta * tf.sign(tf.random_uniform(
                [self._num_samples] + x_shape[1:],
                minval=-1., maxval=1., dtype=tf_dtype))
        return delta_x

    def _compute_gradients(self, loss_fn, x, unused_optim_state):
        """Compute gradient estimates using SPSA."""
        # Assumes `x` is a list, containing a [1, H, W, C] image
        assert len(x) == 1 and x[0].get_shape().as_list()[0] == 1
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
                cond, body,
                loop_vars=[0, tf.TensorArray(size=self._num_iters,
                           dtype=tf_dtype)],
                back_prop=False, parallel_iterations=1)
        avg_grad = tf.reduce_sum(all_grads.stack(), axis=0)
        return [avg_grad]


def _project_perturbation(perturbation, epsilon, input_image):
    """Project `perturbation` onto L-infinity ball of radius `epsilon`."""
    clipped_perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
    new_image = tf.clip_by_value(input_image + clipped_perturbation, 0., 1.)
    return new_image - input_image


def pgd_attack(loss_fn, input_image, label, epsilon, num_steps,
               optimizer=UnrolledAdam(),
               project_perturbation=_project_perturbation,
               early_stop_loss_threshold=None,
               is_debug=False):
    """Projected gradient descent for generating adversarial images.

    Args:
        :param loss_fn: A callable which takes `input_image` and `label` as
                        arguments, and returns a batch of loss values. Same
                        interface as UnrolledOptimizer.
        :param input_image: Tensor, a batch of images
        :param label: Tensor, a batch of labels
        :param epsilon: float, the L-infinity norm of the maximum allowable
                                        perturbation
        :param num_steps: int, the number of steps of gradient descent
        :param optimizer: An `UnrolledOptimizer` object
        :param project_perturbation: A function, which will be used to enforce
                                     some constraint. It should have the same
                                     signature as `_project_perturbation`.
        :param early_stop_loss_threshold: A float or None. If specified, the
                                          attack will end if the loss is below
                                          `early_stop_loss_threshold`.
        :param is_debug: A bool. If True, print debug info for attack progress.

    Returns:
        adversarial version of `input_image`, with L-infinity difference less
            than epsilon, which tries to minimize loss_fn.

    Note that this function is not intended as an Attack by itself. Rather, it
    is designed as a helper function which you can use to write your own attack
    methods. The method uses a tf.while_loop to optimize a loss function in
    a single sess.run() call.
    """

    init_perturbation = tf.random_uniform(tf.shape(input_image),
                                          minval=-epsilon, maxval=epsilon,
                                          dtype=tf_dtype)
    init_perturbation = project_perturbation(init_perturbation,
                                             epsilon, input_image)
    init_optim_state = optimizer.init_state([init_perturbation])
    nest = tf.contrib.framework.nest

    def loop_body(i, perturbation, flat_optim_state):
        """Update perturbation to input image."""
        optim_state = nest.pack_sequence_as(structure=init_optim_state,
                                            flat_sequence=flat_optim_state)

        def wrapped_loss_fn(x):
            return loss_fn(input_image + x, label)
        new_perturbation_list, new_optim_state = optimizer.minimize(
                wrapped_loss_fn, [perturbation], optim_state)
        loss = tf.reduce_mean(wrapped_loss_fn(perturbation), axis=0)
        if is_debug:
            with tf.device("/cpu:0"):
                loss = tf.Print(loss, [loss], "Total batch loss")
        projected_perturbation = project_perturbation(
                new_perturbation_list[0], epsilon, input_image)
        with tf.control_dependencies([loss]):
            i = tf.identity(i)
            if early_stop_loss_threshold:
                i = tf.cond(tf.less(loss, early_stop_loss_threshold),
                            lambda: float(num_steps), lambda: i)
        return i + 1, projected_perturbation, nest.flatten(new_optim_state)

    def cond(i, *_):
        return tf.less(i, num_steps)

    flat_init_optim_state = nest.flatten(init_optim_state)
    _, final_perturbation, _ = tf.while_loop(
        cond,
        loop_body,
        loop_vars=[tf.constant(0.), init_perturbation,
                   flat_init_optim_state],
        parallel_iterations=1,
        back_prop=False)

    if project_perturbation == _project_perturbation:
        check_diff = tf.assert_less_equal(final_perturbation, epsilon * 1.1)
    else:
        check_diff = tf.no_op()
    with tf.control_dependencies([check_diff]):
        adversarial_image = input_image + final_perturbation
    return tf.stop_gradient(adversarial_image)


def margin_logit_loss(model_logits, label, num_classes=10):
    """Computes difference between logit for `label` and next highest logit.

    The loss is high when `label` is unlikely (targeted by default).
    This follows the same interface as `loss_fn` for UnrolledOptimizer and
    pgd_attack, i.e. it returns a batch of loss values.
    """
    logit_mask = tf.one_hot(label, depth=num_classes, axis=-1)
    label_logits = tf.reduce_sum(logit_mask * model_logits, axis=-1)
    logits_with_target_label_neg_inf = model_logits - logit_mask * 99999
    highest_nonlabel_logits = tf.reduce_max(
        logits_with_target_label_neg_inf, axis=-1)
    loss = highest_nonlabel_logits - label_logits
    return loss
