import keras

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval
from cleverhans.attacks import fgsm

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
    """
    MNIST cleverhans tutorial
    :return:
    """

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print "Created TensorFlow session and set Keras backend."

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    print "Loaded MNIST test data."

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

    # Define TF model graph
    model = model_mnist()
    predictions = model(x)
    print "Defined TensorFlow model graph."

    # Train an MNIST model
    tf_model_train(sess, x, y, predictions, X_train, Y_train)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    assert X_test.shape[0] == 10000, X_test.shape
    print 'Test accuracy on legitimate test examples: ' + str(accuracy)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
    print'Test accuracy on adversarial examples: ' + str(accuracy)

    print "Repeating the process, using adversarial training"
    # Redefine TF model graph
    model_2 = model_mnist()
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.3)
    predictions_2_adv = model_2(adv_x_2)

    # Perform adversarial training
    tf_model_train(sess, x, y, predictions_2, X_train, Y_train, predictions_adv=predictions_2_adv)

    # Evaluate the accuracy of the adversarialy trained MNIST model on
    # legitimate test examples
    accuracy = tf_model_eval(sess, x, y, predictions_2, X_test, Y_test)
    print 'Test accuracy on legitimate test examples: ' + str(accuracy)

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM) on
    # the new model, which was trained using adversarial training
    X_test_adv_2, = batch_eval(sess, [x], [adv_x_2], [X_test])
    assert X_test_adv_2.shape[0] == 10000, X_test_adv_2.shape

    # Evaluate the accuracy of the adversarially trained MNIST model on
    # adversarial examples
    accuracy_adv = tf_model_eval(sess, x, y, predictions_2, X_test_adv_2, Y_test)
    print'Test accuracy on adversarial examples: ' + str(accuracy_adv)

    sess.close()

if __name__ == '__main__':
    app.run()
