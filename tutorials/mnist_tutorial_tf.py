from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")

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


    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

    print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = model_mnist()
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.3)
    predictions_2_adv = model_2(adv_x_2)


    def evaluate_2():
        # Evaluate the accuracy of the adversarialy trained MNIST model on
        # legitimate test examples
        accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        accuracy_adv = model_eval(sess, x, y, predictions_2_adv, X_test, Y_test)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

    # Perform adversarial training
    model_train(sess, x, y, predictions_2, X_train, Y_train, predictions_adv=predictions_2_adv,
            evaluate=evaluate_2)


if __name__ == '__main__':
    app.run()
