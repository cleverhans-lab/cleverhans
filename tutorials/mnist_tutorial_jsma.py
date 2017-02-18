from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import numpy as np
import os
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import tf_model_train, model_eval

from cleverhans.attacks import jsma
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes, cnn_model

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
flags.DEFINE_integer('source_samples', 5, 'Nb of test set examples to attack')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


def main(argv=None):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    ###########################################################################
    # Define the dataset and model
    ###########################################################################

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
              "to 'tf', temporarily setting to 'th'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    print("Loaded MNIST test data.")

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model if it does not exist in the train_dir folder
    saver = tf.train.Saver()
    save_path = os.path.join(FLAGS.train_dir, FLAGS.filename)
    if os.path.isfile(save_path):
        saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.filename))
    else:
        train_params = {
            'nb_epochs': FLAGS.nb_epochs,
            'batch_size': FLAGS.batch_size,
            'learning_rate': FLAGS.learning_rate
        }
        model_train(sess, x, y, predictions, X_train, Y_train,
                    args=train_params)
        saver.save(sess, save_path)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(FLAGS.source_samples) + ' * '
          + str(FLAGS.nb_classes) + ' adversarial examples')

    # This array indicates whether an adversarial example was found for each
    # test set sample and target class
    results = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='i')

    # This array contains the fraction of perturbed features for each test set
    # sample and target class
    perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samples),
                             dtype='f')

    # Define the TF graph for the model's Jacobian
    grads = jacobian_graph(predictions, x)

    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, FLAGS.source_samples):
        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(Y_test[sample_ind]))
        target_classes = other_classes(FLAGS.nb_classes, current_class)

        # Loop over all target classes
        for target in target_classes:
            print('--------------------------------------')
            print('Creating adv. example for target class ' + str(target))

            # This call runs the Jacobian-based saliency map approach
            _, res, percent_perturb = jsma(sess, x, predictions, grads,
                                           X_test[sample_ind:(sample_ind+1)],
                                           target, theta=1, gamma=0.1,
                                           increase=True, back='tf',
                                           clip_min=0, clip_max=1)

            # Update the arrays for later analysis
            results[target, sample_ind] = res
            perturbations[target, sample_ind] = percent_perturb

    # Compute the number of adversarial examples that were successfuly found
    nb_targets_tried = ((FLAGS.nb_classes - 1) * FLAGS.source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.2f}'.format(succ_rate))

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.2f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.2f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()

if __name__ == '__main__':
    app.run()
