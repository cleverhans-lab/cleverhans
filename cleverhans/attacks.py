import copy
import keras
import math
import numpy as np
import tensorflow as tf

from utils import batch_indices
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
