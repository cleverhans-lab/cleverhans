import copy
import itertools
import keras
import numpy as np
import tensorflow as tf
import multiprocessing as mp

import utils_tf

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def fgsm(x, predictions, eps, back='tf', clip_min=None, clip_max=None):
    """
    A wrapper for the Fast Gradient Sign Method.
    It calls the right function, depending on the
    user's backend.
    :param x: the input
    :param predictions: the model's output
    :param eps: the epsilon (input variation parameter)
    :param back: switch between TensorFlow ('tf') and
                Theano ('th') implementation
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    if back == 'tf':
        # Compute FGSM using TensorFlow
        return fgsm_tf(x, predictions, eps, clip_min=None, clip_max=None)
    elif back == 'th':
        raise NotImplementedError("Theano FGSM not implemented.")

def fgsm_tf(x, predictions, eps, clip_min=None, clip_max=None):
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
    y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss = utils_tf.tf_model_loss(y, predictions, mean=False)

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

def saliency(sess, x, f_x, sample,
             target, theta, gamma=np.inf,
             clip_min=None, clip_max=None,
             increase=True, back='tf'):
    """
    A wrapper for the saliency method algorithm. It calls the right
    function, depending on the user's backend.
    :param sess: The TensorFlow session
    :param x: The TensorFlow placeholder for the input
    :param f_x: The symbolic output of the model. NOTE: this is the linear output,
    pre-softmax.
    :param sample: The input (1 x 1 x n_rows x n_cols) image sample
    :param target: The target label for this sample
    :param theta: The delta for each feature adjustment
    :param gamma: A float between 0 - 1 indicating the maximum distortion percentage
    :param clip_min: The minimum allowed feature value
    :param clip_max: The maximum allowed feature value
    :param increase: Boolean; true if we are increasing pixels, false otherwise.
    """
    if back == 'tf':
        # Compute FGSM using TensorFlow
        return saliency_tf(sess, x, f_x, sample, target,
                           theta, gamma, clip_min,
                           clip_max, increase)
    elif back == 'th':
        raise NotImplementedError("Theano saliency method not implemented.")

def compute_saliency(x):
    """
    A helper function for saliency_tf. This is used for a parallelized
    map() operation via multiprocessing.Pool().
    :param x: A tuple containing (point, point, grads_target,
    grads_others, increase).
    :return: The saliency value for this pair of points. Either -alpha*beta
    if conditions are met, or 0 otherwise.
    """
    p, q, grads_target, grads_others, increase = x
    alpha = grads_target[p[0], p[1]] + grads_target[q[0], q[1]]
    beta = grads_others[p[0], p[1]] + grads_others[q[0], q[1]]
    if (increase and alpha > 0 and beta < 0) or (not increase and alpha < 0 and beta > 0):
        return -alpha * beta
    else:
        return 0

def model_argmax(sess, x, f_x, X):
    """
    A helper function for saliency_tf. Computes the current
    model argmax output (the current class prediction).
    :param sess: The TensorFlow session
    :param x: The tensorflow placeholder for the input
    :param f_x: The symbolic output of the model
    :param X: The input (1 x 1 x n_rows x n_cols) image sample
    :return: The argmax output of f_x (the model's current predicted class)
    """
    s = tf.argmax(f_x, dimension=1)
    feed_dict = {x: X, keras.backend.learning_phase(): 0}
    return sess.run(s, feed_dict)[0]

def saliency_map(sess, x, f_x, X, target, search_domain, increase):
    """
    A helper function for saliency_tf. This function implements Algorithm 2 from
    Papernot's paper.
    :param sess: The TensorFlow session
    :param x: The tensorflow placeholder for the input
    :param f_x: The symbolic output of the model
    :param X: The input (1 x 1 x n_rows x n_cols) image sample
    :param target: The target label for this sample
    :param search_domain: the pairs of points that remain to be considered
    :param increase: Boolean; true if we are increasing pixels, false otherwise.
    """
    # compute derivatives for target class
    deriv_target, = tf.gradients(f_x[:, target], x)
    # compute derivatives for other classes
    other_classes = [i for i in range(FLAGS.nb_classes) if i != target]
    deriv_others, = tf.gradients([f_x[:, i] for i in other_classes], x)
    grads_target, grads_others = \
        sess.run([tf.reshape(deriv_target, (28, 28)), tf.reshape(deriv_others, (28, 28))],
                 {x: X, keras.backend.learning_phase(): 0})
    pool = mp.Pool()
    outs = pool.map(compute_saliency, [(p, q, grads_target, grads_others, increase) \
                                    for p, q in itertools.combinations(search_domain, 2)])
    ind = np.argmax(outs)
    pairs = [elt for elt in itertools.combinations(search_domain, 2)]
    return pairs[ind]

def saliency_tf(sess, x, f_x, sample,
                target, theta, gamma,
                clip_min, clip_max,
                increase):
    """
    :param sess: The TensorFlow session
    :param x: The TensorFlow placeholder for the input
    :param f_x: The symbolic output of the model
    :param sample: The input (1 x 1 x n_rows x n_cols) image sample
    :param target: The target label for this sample
    :param theta: The delta for each feature adjustment
    :param gamma: A float between 0 - 1 indicating the maximum distortion percentage
    :param clip_min: The minimum allowed feature value
    :param clip_max: The maximum allowed feature value
    :param increase: Boolean; true if we are increasing pixels, false otherwise.
    """
    X = copy.copy(sample)
    max_iter = np.floor(np.product(X[0][0].shape) * gamma / 2)
    print('Maximum # of iterations: %i' % max_iter)
    if increase:
        # Since we'll be increasing pixel values, our search domain must contain
        # only pixels that are not already maxed out.
        search_domain = set([(row, col) for row in xrange(FLAGS.img_rows) \
                             for col in xrange(FLAGS.img_cols) if X[0, 0, row, col] < clip_max])
    else:
        # Since we'll be decreasing pixel values, our search domain must contain
        # only pixels that are not at the minimum.
        search_domain = set([(row, col) for row in xrange(FLAGS.img_rows) \
                             for col in xrange(FLAGS.img_cols) if X[0, 0, row, col] > clip_min])
    # compute model argmax
    s = model_argmax(sess, x, f_x, X)
    print('Start class: %i' % s)
    # begin loop
    i = 0
    while s != target and i < max_iter and len(search_domain) > 0:
        if i%5 == 0:
            print('iteration # %i' % i)
        # find pixels to change, update them
        p1, p2 = saliency_map(sess, x, f_x, X, target, search_domain, increase)
        # sanity check p1 and p2
        assert 0 <= p1[0] < FLAGS.img_rows and 0 <= p2[0] < FLAGS.img_rows
        assert 0 <= p1[1] < FLAGS.img_cols and 0 <= p2[1] < FLAGS.img_cols
        # update pixel values at p1 and p2
        if increase:
            X[0, 0, p1[0], p1[1]] = np.minimum(clip_max, X[0, 0, p1[0], p1[1]] + theta)
            X[0, 0, p2[0], p2[1]] = np.minimum(clip_max, X[0, 0, p2[0], p2[1]] + theta)
        else:
            X[0, 0, p1[0], p1[1]] = np.maximum(clip_min, X[0, 0, p1[0], p1[1]] - theta)
            X[0, 0, p2[0], p2[1]] = np.maximum(clip_min, X[0, 0, p2[0], p2[1]] - theta)
        assert np.max(X) <= clip_max
        assert np.min(X) >= clip_min
        # remove pixels from search domain
        l = len(search_domain)
        if X[0, 0, p1[0], p1[1]] == clip_min or X[0, 0, p1[0], p1[1]] == clip_max:
            search_domain.remove(p1)
            assert len(search_domain) < l
        if X[0, 0, p2[0], p2[1]] == clip_min or X[0, 0, p2[0], p2[1]] == clip_max:
            search_domain.remove(p2)
            assert len(search_domain) < l
        # update argmax
        s = model_argmax(sess, x, f_x, X)
        if i % 5 == 0:
            print('Current class: %i' % s)
        i += 1

    return X
