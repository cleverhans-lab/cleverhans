import math
import os
import keras
import tensorflow as tf

from attacks import fgsm
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


def tf_model_train(sess, x, y, model, X_train, Y_train, save=False, adversarial=False):
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
    loss = tf_model_loss(y, model)

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

            for batch in range(nb_batches):
                if batch % 100 == 0 and batch > 0:
                    print("Batch " + str(batch))

                # Compute batch start and end indices
                start, end = batch_indices(batch, len(X_train), FLAGS.batch_size)

                # Perform one training step
                train_step.run(feed_dict={x: X_train[start:end],
                                          y: Y_train[start:end],
                                          keras.backend.learning_phase(): 1})

                if adversarial:
                    # Compute adversarial examples
                    adv_ex = fgsm(sess, x, y, model, X_train[start:end], Y_train[start:end], 0.3)

                    # Train on adversarial examples
                    train_step.run(feed_dict={x: adv_ex,
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
