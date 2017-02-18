from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import copy
import itertools
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from six.moves import xrange

from . import utils_tf
from . import utils

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def fgsm(x, predictions, eps, clip_min=None, clip_max=None):
    """
    TensorFlow implementation of the Fast Gradient
    Sign method.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """

    # Compute loss
    y = tf.to_float(
        tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss = utils_tf.model_loss(y, predictions, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
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


def saliency_score(packed_data):
    """
    Helper function for saliency_map. This is used for a parallelized map()
    operation via multiprocessing.Pool()
    :param packed_data: tuple containing (index, index, gradients, target,
    other_classes, increase).
    : return: saliency score for the pair of indices i, j. Either
    target_sum * abs(other_sum) if the conditions are met, or 0 otherwise.
    """

    # compute the saliency score for the given pair
    i, j, grads_target, grads_others, increase = packed_data
    target_sum = grads_target[i] + grads_target[j]
    other_sum = grads_others[i] + grads_others[j]

    # evaluate the saliency map conditions
    if ((increase and target_sum > 0 and other_sum < 0) or
       (not increase and target_sum < 0 and other_sum > 0)):
        return -target_sum * other_sum
    else:
        return 0


def saliency_map(grads_target, grads_other, search_domain, increase):
    """
    TensorFlow implementation for computing salency maps
    :param grads_target: a matrix containing forward derivatives for the
    target class
    :param grads_other: a matrix where every element is the sum of forward
    derivatives over all non-target classes at that index
    :param search_domain: the set of input indices that we are considering
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :return: (i, j, search_domain) the two input indices selected and the
             updated search domain
    """

    # determine the saliency score for every pair of pixels from our search
    # domain
    pool = mp.Pool()
    scores = pool.map(saliency_score,
                      [(i, j, grads_target, grads_other, increase)
                       for i, j in itertools.combinations(search_domain, 2)])

    # wait for the threads to finish to free up memory
    pool.close()
    pool.join()

    # grab the pixels with the largest scores
    candidates = np.argmax(scores)
    pairs = [elt for elt in itertools.combinations(search_domain, 2)]

    # update our search domain
    search_domain.remove(pairs[candidates][0])
    search_domain.remove(pairs[candidates][1])

    return pairs[candidates][0], pairs[candidates][1], search_domain


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
    if 'keras' in sys.modules:
        import keras
        feed_dict = {x: X, keras.backend.learning_phase(): 0}
    else:
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
    :return:
    """
    # This function will return a list of TF gradients
    list_derivatives = []

    # Define the TF graph elements to compute our derivatives for each class
    for class_ind in xrange(nb_classes):
        derivatives, = tf.gradients(predictions[:, class_ind], x)
        list_derivatives.append(derivatives)

    return list_derivatives


def jsma_tf(sess, x, predictions, grads, sample, target, theta, gamma,
            increase, clip_min, clip_max):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param sample: numpy array with sample input
    :param target: target class for sample input
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
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
    print('Maximum number of iterations: {0}'.format(max_iters))

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
           len(search_domain) > 0):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_original_shape,
                                              nb_features, FLAGS.nb_classes)

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

        # This process may take a while, so outputting progress regularly
        if iteration % 5 == 0:
            msg = 'Current iteration: {0} - Current Prediction: {1}'
            print(msg.format(iteration, current))

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        print('Successful')
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        print('Unsuccesful')
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


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
    :param keras_phase: if not None, contains keras.backend.learning_phase()
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
    for ind, input in enumerate(X_sub_prev):
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Prepare feeding dictionary
        feed_dict = {x: np.reshape(input, input_shape)}
        if keras_phase is not None:
            feed_dict[keras_phase] = 0

        # Compute sign matrix
        grad_val = sess.run([tf.sign(grad)], feed_dict=feed_dict)[0]

        # Create new synthetic point in adversary substitute training set
        X_sub[2*ind] = X_sub[ind] + lmbda * grad_val

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub
