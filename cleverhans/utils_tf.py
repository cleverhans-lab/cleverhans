from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
import os
import keras
import six
import tensorflow as tf
import time

from .utils import batch_indices

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def tf_model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out


def tf_model_train(sess, model, X_train, Y_train, save=False,
                   evaluate=None, adversarial_training=False,
                   adv_eps=None, adv_clip_min=None,
                   adv_clip_max=None):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param model: the model graph
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: Boolean controlling the save operation
    :param evaluate: a function for evaluating the model at the end of
    each training epoch
    :param adversarial_training: Boolean controlling whether or not we will
    perform adversarial training
    :param adv_eps:
    :param adv_clip_min:
    :param adv_clip_max:
    :return: True if model trained
    """
    print("Starting model training using TensorFlow.")

    if adversarial_training:
        assert adv_eps is not None

    # Create placeholders for inputs and labels
    x = tf.placeholder(tf.float32, shape=(None,) + X_train.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y_train.shape[1:])

    # Define loss
    loss = tf_model_loss(y, model(x))
    if adversarial_training:
        from .attacks import fgsm_tf_symbolic
        x_adv = fgsm_tf_symbolic(x, model, adv_eps, adv_clip_min, adv_clip_max)
        loss = (loss + tf_model_loss(y, model(x_adv))) / 2

    train_step = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, rho=0.95, epsilon=1e-08).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
    print("Defined optimizer.")

    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)

        for epoch in six.moves.xrange(FLAGS.nb_epochs):
            print("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / FLAGS.batch_size))
            assert nb_batches * FLAGS.batch_size >= len(X_train)

            prev = time.time()
            for batch in range(nb_batches):

                # Compute batch start and end indices
                start, end = batch_indices(batch, len(X_train), FLAGS.batch_size)

                # Perform one training step
                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end],
                                          keras.backend.learning_phase(): 1})
            assert end >= len(X_train) # Check that all examples were used
            cur = time.time()
            print("\tEpoch took " + str(cur - prev) + " seconds")
            prev = cur
            if evaluate is not None:
                evaluate()


        if save:
            save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print("Completed model training and model saved at:" + str(save_path))
        else:
            print("Completed model training.")

    return True


def tf_model_eval(sess, model, X_test, Y_test):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param model: the model graph
    :return: a float with the accuracy value
    """
    # Create placeholders for inputs and labels
    x = tf.placeholder(tf.float32, shape=(None,) + X_test.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y_test.shape[1:])

    # Define sympbolic for accuracy
    acc_value = keras.metrics.categorical_accuracy(y, model(x))

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / FLAGS.batch_size))
        assert nb_batches * FLAGS.batch_size >= len(X_test)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * FLAGS.batch_size
            end = min(len(X_test), start + FLAGS.batch_size)
            cur_batch_size = end - start

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            accuracy += cur_batch_size * acc_value.eval(feed_dict={x: X_test[start:end],
                                            y: Y_test[start:end],
                                            keras.backend.learning_phase(): 0})
        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy


def tf_model_load(sess):
    """

    :param sess:
    :param x:
    :param y:
    :param model:
    :return:
    """
    with sess.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.filename))

    return True

def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    """
    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in six.moves.xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in six.moves.xrange(0, m, FLAGS.batch_size):
            batch = start // FLAGS.batch_size
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Compute batch start and end indices
            start = batch * FLAGS.batch_size
            end = start + FLAGS.batch_size
            numpy_input_batches = [numpy_input[start:end] for numpy_input in numpy_inputs]
            cur_batch_size = numpy_input_batches[0].shape[0]
            assert cur_batch_size <= FLAGS.batch_size
            for e in numpy_input_batches:
                assert e.shape[0] == cur_batch_size

            feed_dict = dict(zip(tf_inputs, numpy_input_batches))
            feed_dict[keras.backend.learning_phase()] = 0
            numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
            for e in numpy_output_batches:
                assert e.shape[0] == cur_batch_size, e.shape
            for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
                out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def model_argmax(sess, x, predictions, sample):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param sample: (1 x 1 x img_rows x img_cols) numpy array with sample input
    :return: the argmax output of predictions, i.e. the current predicted class
    """

    feed_dict = {x: sample, keras.backend.learning_phase(): 0}
    probabilities = sess.run(predictions, feed_dict)

    return np.argmax(probabilities)
