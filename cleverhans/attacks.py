from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import itertools
import keras
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from .utils import batch_indices
from . import utils_tf 

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

import multiprocessing as mp
import resource


def jsma(sess, x, predictions, sample, target, theta, gamma=np.inf, increase=True, back='tf', clip_min=None, clip_max=None):
    """
    A wrapper for the Jacobian-based saliency map approach.
    It calls the right function, depending on the
    user's backend.
    :param sess: TF session
    :param x: the input
    :param predictions: the model's symbolic output (linear output, pre-softmax)
    :param sample: (1 x 1 x img_rows x img_cols) numpy array with sample input
    :param target: target class for input sample
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indiciating the maximum distortion percentage
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param back: switch between TensorFlow ('tf') and
                Theano ('th') implementation
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: an adversarial sample
    """
    if back == 'tf':
        # Compute Jacobian-based saliency map attack using TensorFlow
        return PLACEHOLDER_tf(sess, x, predictions, sample, target, theta, gamma, increase, clip_min, clip_max)
    elif back == 'th':
        raise NotImplementedError("Theano PLACEHOLDER not implemented.")

def model_argmax(sess, x, predictions, sample):
    """
    Helper function for jsma_tf that computes the current class prediction
    :param sess: TF session 
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param sample: (1 x 1 x img_rows x img_cols) numpy array with sample input
    :return: the argmax output of predictions, i.e. the current predicted class
    """

    feed_dict = {x: sample, keras.backend.learning_phase(): 0}
    probabilities = sess.run(predictions, feed_dict)

    return np.argmax(probabilities)

def apply_perturbations(i, j, X, increase, theta, clip_min, clip_max):
    """
    TensorFlow implementation for apply perterbations to input features based on salency maps
    :param i: row of our selected pixel
    :param j: column of our selected pixel
    :param X: a matrix containing our input features for our sample
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param theta: delta for each feature adjustment
    :param clip_min: mininum value for a feature in our sample
    :param clip_max: maximum value for a feature in our sample
    : return: a perterbed input feature matrix for a target class
    """

    # perterb our input sample
    if increase:
        X[0, 0, i[0], i[1]] = np.minimum(clip_max, X[0, 0, i[0], i[1]] + theta)
        X[0, 0, j[0], j[1]] = np.minimum(clip_max, X[0, 0, j[0], j[1]] + theta)

    return X

def saliency_score(packed_data):
    """
    Helper function for saliency_map. This is used for a parallelized map() operation
    via multiprocessing.Pool()
    :param packed_data: tuple containing (point, point, gradients, target, 
    other_classes, increase).
    : return: saliency score for the pair of points i, j. Either target_sum * abs(other_sum)
    if the conditions are met, or 0 otherwise.
    """

    # compute the saliency score for the given pair
    i, j, grads_target, grads_others, increase = packed_data
    target_sum = grads_target[i[0],i[1]] + grads_target[j[0],j[1]]
    other_sum = grads_others[i[0],i[1]] + grads_others[j[0],j[1]]

    # evaluate the saliency map conditions
    if (increase and target_sum > 0 and other_sum < 0) or (not increase and target_sum < 0 and other_sum > 0):
        return -target_sum * other_sum
    else:
        return 0

def saliency_map(grads_target, grads_other, search_domain, increase):
    """
    TensorFlow implementation for computing salency maps
    :param jacobian: a matrix containing forward derivatives for all classes
    :param target: the desired target class for the sample
    : return: a vector of scores for the target class
    """

    # determine the saliency score for every pair of pixels from our search domain
    pool = mp.Pool()
    scores = pool.map(saliency_score, [(i, j, grads_target, grads_other, increase) \
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

def jacobian(sess, x, predictions, deriv_target, deriv_others, X):
    """
    TensorFlow implementation of the foward derivative
    :param x: the input placeholder
    :param X: numpy array with sample input
    :param predictions: the model's symbolic output
    :return: matrix of forward derivatives flattened into vectors
    """

    # compute the gradients for all classes
    grad_target, grad_others = \
            sess.run([tf.reshape(deriv_target, (FLAGS.img_rows, FLAGS.img_cols)),
                tf.reshape(deriv_others, (FLAGS.img_rows, FLAGS.img_cols))], {x: X, keras.backend.learning_phase(): 0})

    return grad_target, grad_others

def jsma_tf(sess, x, predictions, sample, target, theta, gamma, increase, clip_min, clip_max):
    """
    TensorFlow implementation of the jsma.
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output (linear output, pre-softmax)
    :param sample: numpy array with sample input
    :param target: target class for sample input
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indiciating the maximum distortion percentage
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: an adversarial sample
    """

    adv_x = copy.copy(sample)
    max_iters = np.floor(np.product(adv_x[0][0].shape) * gamma / 2)
    print('Maximum number of iterations: {0}'.format(max_iters))

    # prep our derivatives for all classes
    deriv_target, = tf.gradients(predictions[:,target], x)
    other_classes = [i for i in range(FLAGS.nb_classes) if i != target]
    deriv_others, = tf.gradients([predictions[:,i] for i in other_classes], x)

    # compute our search domain based on maximizing or minimizing pixels
    if increase:
        search_domain = set([(row, col) for row in xrange(FLAGS.img_rows) \
                 for col in xrange(FLAGS.img_cols) if adv_x[0, 0, row, col] < clip_max])
    else:
        search_domain = set([(row, col) for row in xrange(FLAGS.img_rows) \
                 for col in xrange(FLAGS.img_cols) if adv_x[0, 0, row, col] > clip_min])

    # repeat until we have achieved misclassification
    iteration = 0
    current = model_argmax(sess, x, predictions, adv_x)	
    while current != target and iteration < max_iters and len(search_domain) > 0: 

        # compute the Jacobian derivatives
        grads_target, grads_others = jacobian(sess, x, predictions, deriv_target, deriv_others, adv_x)

        # compute the salency map for each of our taget classes
        i, j, search_domain = saliency_map(grads_target, grads_others, search_domain, increase)

        # apply an adversarial perterbation to the sample
        adv_x = apply_perturbations(i, j, adv_x, increase, theta, clip_min, clip_max)

        # update our current prediction
        current = model_argmax(sess, x, predictions, adv_x)
        iteration = iteration + 1

        if iteration % 5 == 0:
            print('Current iteration: {0} - Current Prediction: {1}'.format(iteration, current))

    percent_perterbed = float(iteration * 2)/float(FLAGS.img_rows * FLAGS.img_cols)
    # failed to perterb the input sample to the target class within the constraints
    if iteration == max_iters or len(search_domain) == 0:
        print 'Unsuccesful'
        return adv_x, -1, percent_perterbed
    # success!
    else:
        print 'Successful'
        return adv_x, 1, percent_perterbed

