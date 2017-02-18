from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import cnn_model
from cleverhans.utils_tf import model_train, model_eval

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
flags.DEFINE_integer('nb_filters', 4, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
    """
    Test the accuracy of the MNIST cleverhans tutorial model
    :return:
    """
    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    with tf.Session() as sess:
        keras.backend.set_session(sess)
        print("Created TensorFlow session and set Keras backend.")

        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist()
        print("Loaded MNIST test data.")

        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

        # Define TF model graph
        model = cnn_model(nb_filters=FLAGS.nb_filters)
        predictions = model(x)
        print("Defined TensorFlow model graph.")

        # Train an MNIST model
        model_train(sess, x, y, predictions, X_train, Y_train)

        print("keras version: ", keras.__version__)
        print("tf version: ", tf.__version__)

        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test)
        assert float(accuracy) >= 0.8, accuracy

def test_mnist_accuracy():
    main()

if __name__ == "__main__":
    main()
