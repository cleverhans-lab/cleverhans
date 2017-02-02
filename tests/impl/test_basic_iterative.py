from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks_tf import basic_iterative

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

class TestBasicIterative(unittest.TestCase):
    """
    A small unit test
    """

    def setUp(self):
        # Image dimensions ordering should follow the Theano convention
        if keras.backend.image_dim_ordering() != 'tf':
            keras.backend.set_image_dim_ordering('tf')
            print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
                  "to 'th', temporarily setting to 'tf'")

        # Create TF session and set as Keras backend session
        self.sess = tf.Session()
        keras.backend.set_session(self.sess)

        # Get MNIST test data
        X_train, Y_train, self.X_test, self.Y_test = data_mnist()

        label_smooth = .1
        Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

        # Define input TF placeholder
        self.x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.y = tf.placeholder(tf.float32, shape=(None, 10))

        # Define TF model graph
        self.model = model_mnist()
        self.predictions = self.model(self.x)
        print("Defined TensorFlow model graph.")

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test examples
            accuracy = model_eval(self.sess, self.x, self.y, self.predictions, self.X_test, self.Y_test)
            print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Define TF model graph
        self.model = model_mnist()
        self.predictions = self.model(self.x)
        # Train the MNIST model
        model_train(self.sess, self.x, self.y, self.predictions, X_train, Y_train, evaluate=evaluate)


    def testBasicIterativeMethod(self):
        # check to make sure the model has a high accuracy
        accuracy = model_eval(self.sess, self.x, self.y, self.predictions, self.X_test, self.Y_test)
        self.assertGreater(accuracy, 0.98)
        print('Test accuracy on normal test examples: ' + str(accuracy))
        print('Computing adversarial samples via basic iterative method...')
        eps = 0.3
        n_iter = 10
        adv_x = basic_iterative(self.x, self.y, self.model, eps=eps, eps_iter=eps/n_iter,
                                n_iter=n_iter, clip_min=0., clip_max=1.)
        X_test_adv, = batch_eval(self.sess, [self.x, self.y], [adv_x], [self.X_test, self.Y_test])
        # check that clip_min and clip_max worked
        self.assertLessEqual(X_test_adv.max(), 1.)
        self.assertGreaterEqual(X_test_adv.min(), 0.)
        # check that we didn't move more than eps away from the original
        max_diff = np.abs(X_test_adv - self.X_test).max()
        self.assertLess(max_diff, eps + 1e-4)
        # check that our accuracy is no greater than 1% on adversarial samples
        accuracy = model_eval(self.sess, self.x, self.y, self.predictions, X_test_adv, self.Y_test)
        self.assertLess(accuracy, 0.01)
        print('Test accuracy on adversarial test examples: ' + str(accuracy))


if __name__ == '__main__':
    unittest.main()
