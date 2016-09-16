import copy
import keras
import math
import numpy as np
import tensorflow as tf

from utils import batch_indices
import utils_tf

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def fgsm(x, predictions, eps, back='tf'):
    """
    This is a wrapper for FGSM that we can use to select between TF and Theano
    :param sess:
    :param x:
    :param y:
    :param model:
    :param X_test:
    :param Y_test:
    :param eps:
    :param back:
    :return:
    """
    if back == 'tf':
        # Compute FGSM using TensorFlow
        return fgsm_tf(x, predictions, eps)
    elif back == 'th':
        raise NotImplementedError("Theano FGSM not implemented.")

def fgsm_tf(x, predictions, eps):
    # Define loss

    y = tf.to_float(tf.equal(predictions, tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    loss = utils_tf.tf_model_loss(y, predictions, mean=False)

    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    scaled_signed_grad = eps * signed_grad
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    return adv_x
