import copy
import keras
import math
import numpy as np
import tensorflow as tf

from utils import batch_indices
import utils_tf

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def fgsm(sess, x, y, model, X_test, Y_test, eps, back='tf'):
    """

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
        return fgsm_tf(sess, x, y, model, X_test, Y_test, eps)
    elif back == 'th':
        # TODO(implement)
        print("Theano FGSM not implemented")


def fgsm_tf(sess, x, y, model, X_test, Y_test, eps):
    # Define loss
    loss = utils_tf.tf_model_loss(y, model, mean=False)

    # Define FGSM
    fgsm_symb = tf.gradients(loss, x)

    # Initial result variable
    result = copy.copy(X_test)

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(len(X_test) / FLAGS.batch_size))

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Compute batch start and end indices
            start, end = batch_indices(batch, len(X_test), FLAGS.batch_size)

            # Compute fast gradient sign direction
            fgsm_val = sess.run(fgsm_symb, feed_dict={x: X_test[start:end],
                                                            y: Y_test[start:end],
                                                            keras.backend.learning_phase(): 0})[0]

            # Add to legitimate samples to create adversarial examplesZZ
            result[start:end] += eps * np.sign(fgsm_val)

    return result