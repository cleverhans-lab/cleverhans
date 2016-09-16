import math
import numpy as np
import os
import keras
import tensorflow as tf
import time

import attacks
from keras.backend import categorical_crossentropy
from tensorflow.python.platform import flags
from utils import batch_indices

FLAGS = flags.FLAGS


def tf_model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """
    if mean:
        # Return mean of the loss
        return tf.reduce_mean(categorical_crossentropy(y, model))
    else:
        # Return a vector with the loss per sample
        return categorical_crossentropy(y, model)


def tf_model_train(sess, x, y, predictions, X_train, Y_train, save=False,
                   predictions_adv=None):
    """
    Train a TF graph
    :param sess:
    :param x:
    :param y:
    :param model:
    :param X_train:
    :param Y_train:
    :param save:
    :return:
    """
    print "Starting model training using TensorFlow."

    # Define loss
    loss = tf_model_loss(y, predictions)
    if predictions_adv is not None:
        loss = (loss + tf_model_loss(y, predictions_adv)) / 2

    train_step = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate, rho=0.95, epsilon=1e-08).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
    print "Defined optimizer."

    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)

        for epoch in xrange(FLAGS.nb_epochs):
            print("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(len(X_train) / FLAGS.batch_size))

            prev = time.time()
            for batch in range(nb_batches):
                if batch % 100 == 0 and batch > 0:
                    print("Batch " + str(batch))
                    cur = time.time()
                    print("\tTook " + str(cur - prev) + " seconds")
                    prev = cur

                # Compute batch start and end indices
                start, end = batch_indices(batch, len(X_train), FLAGS.batch_size)

                # Perform one training step
                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end],
                                          keras.backend.learning_phase(): 1})


        if save:
            save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            print "Completed model training and model saved at:" + str(save_path)
        else:
            print "Completed model training."

    return True


def tf_model_eval(sess, x, y, model, X_test, Y_test):
    """

    :param sess:
    :param x:
    :param y:
    :param model:
    :param X_test:
    :param Y_test:
    :return:
    """
    # Define sympbolic for accuracy
    acc_value = keras.metrics.categorical_accuracy(y, model)

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(len(X_test) / FLAGS.batch_size))

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Compute batch start and end indices
            start, end = batch_indices(batch, len(X_test), FLAGS.batch_size)

            accuracy += acc_value.eval(feed_dict={x: X_test[start:end],
                                            y: Y_test[start:end],
                                            keras.backend.learning_phase(): 0})

        # Divide by number of batches to get final value
        accuracy /= nb_batches

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
    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in tf_outputs:
        out.append([])
    with sess.as_default():
        for start in xrange(0, m, FLAGS.batch_size):
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

    out = map(lambda x: np.concatenate(x, axis=0), out)
    for e in out:
        assert e.shape[0] == m, e.shape
    return out
