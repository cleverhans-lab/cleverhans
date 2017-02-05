from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm
from cleverhans.attacks_tf import jacobian_graph

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('holdout', 100, 'Test set holdout for adversary')
flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
flags.DEFINE_integer('nb_epochs_s', 6, 'Number of epochs to train substitute')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def setup_tutorial():
    """
    Helper function to check correct configuration of tf and keras for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")

    return True


def prepare_black_box(sess, x, y, X_train, Y_train, X_test, Y_test):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :return:
    """

    # Define TF model graph (for the black-box model)
    model = model_mnist()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Train an MNIST model
    model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate)

    return predictions


def jacobian_augmentation(sess, x, X_sub_prev, Y_sub, grads):
    X_sub_shape = np.shape(X_sub_prev)
    X_sub_shape = (2*X_sub_shape[0], X_sub_shape[1:])
    X_sub = np.zeros(X_sub_shape)

    for ind, input in enumerate(X_sub_prev):
        grad = grads[Y_sub[ind]]
        grad_val = sess.run([tf.sign(grad)], feed_dict={x: input, keras.backend.learning_phase(): 0})[0]
        X_sub[2*ind] = X_sub[ind] + FLAGS.lmbda * grad_val

    return X_sub


def train_substitute(sess, x, y, black_box_predictions, X_sub, Y_sub):

    # Define TF model graph (for the black-box model)
    model_sub = model_mnist()
    predictions_sub = model_sub(x)
    print("Defined TensorFlow model graph.")

    grads = jacobian_graph(predictions_sub, x)

    for rho in xrange(FLAGS.nb_epochs_s):
        model_train(sess, x, y, predictions_sub, X_sub, Y_sub)

        if rho < FLAGS.nb_epochs_s - 1:
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads)
            Y_sub[len(X_sub)/2:] = np.argmax(batch_eval(sess, [x], [black_box_predictions], [X_sub[len(X_sub)/2:]]), axis=1)

    return predictions_sub


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()

    # Initialize substitute training set for adversary
    X_sub = X_test[:FLAGS.holdout]
    Y_sub = Y_test[:FLAGS.holdout]

    # Redefine test set as remaining samples
    X_test = X_test[FLAGS.holdout:]
    Y_test = Y_test[FLAGS.holdout:]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    black_box_predictions = prepare_black_box(sess, x, y, X_train, Y_train, X_test, Y_test)

    # Train substitute
    substitute_predictions = train_substitute(sess, x, y, black_box_predictions, X_sub, Y_sub)

    # Craft adversarial examples using the substitute
    adv_x = fgsm(x, substitute_predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, black_box_predictions, X_test_adv, Y_test)
    print('Test accuracy of oracle on substitute adversarial examples: ' + str(accuracy))


if __name__ == '__main__':
    app.run()
