import keras
import numpy as np
import os

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import tf_model_train, tf_model_eval
from cleverhans.attacks import jsma

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
flags.DEFINE_integer('source_samples', 5, 'Number of examples in test set to attack')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
    """
    MNIST cleverhans tutorial for the Jacobian-based saliency map approach (JSMA)
    :return:
    """
    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print "INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'"

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

    # Train an MNIST model if it does not exist in the train_dir folder
    saver = tf.train.Saver()
    save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
    if os.path.isfile(save_path):
        saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.filename))
    else:
        tf_model_train(sess, x, y, predictions, X_train, Y_train)
        saver.save(sess, save_path)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    # Craft adversarial examples for nb_classes from per_samples using the Jacobian-based saliency map approach 
    results = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='i')
    perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='f')
    print 'Crafting ' + str(FLAGS.source_samples) + ' * ' + str(FLAGS.nb_classes) + ' adversarial examples'

    for sample in xrange(FLAGS.source_samples):
        target_classes = list(xrange(FLAGS.nb_classes))
        target_classes.remove(np.argmax(Y_test[sample]))
        for target in target_classes:
            print '--------------------------------------\nCreating adversarial example for target class ' + str(target)
            _, result, percentage_perterb = jsma(sess, x, predictions, X_test[sample:(sample+1)], target,
                    theta=1, gamma=0.1, increase=True, back='tf', clip_min=0, clip_max=1)
            results[target, sample] = result
            perturbations[target, sample] = percentage_perterb

    success_rate = np.sum(results) / ((FLAGS.nb_classes - 1) * FLAGS.source_samples)
    percentage_perterbed = np.mean(perturbations)

    print('Avg. rate of successful misclassifcations {0} \n avg. rate of perterbed features {1}'\
            .format(success_rate, percentage_perterbed))

if __name__ == '__main__':
    app.run()
