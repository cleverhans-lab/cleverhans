from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks_tf import basic_iterative

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
    """
    MNIST cleverhans iterative FGSM tutorial
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', "
              "temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = model_mnist()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Train an MNIST model
    model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate)

    # Craft adversarial examples using the iterative Fast Gradient Sign Method
    print('Computing adversarial samples via iterative FGSM...')
    eps = 0.3
    n_iter = 10
    adv_x = basic_iterative(x, y, model, eps=eps, eps_iter=eps/n_iter,
                            n_iter=n_iter, clip_min=0., clip_max=1.)
    X_test_adv, = batch_eval(sess, [x, y], [adv_x], [X_test, Y_test])
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape
    # check that we didn't move more than eps away from the original
    max_diff = np.abs(X_test_adv - X_test).max()
    assert max_diff <= eps+1e-4, max_diff

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test)
    print('Test accuracy on adversarial examples: ' + str(accuracy))


if __name__ == '__main__':
    app.run()
