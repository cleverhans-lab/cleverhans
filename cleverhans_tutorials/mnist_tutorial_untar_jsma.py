"""
This tutorial shows how to generate adversarial examples
using JSMA in white-box setting.
The original paper can be found at:
https://arxiv.org/abs/1511.07528
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.loss import LossCrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import train, model_eval, model_argmax
from cleverhans_tutorials.tutorial_models import ModelBasicCNN

FLAGS = flags.FLAGS


def mnist_tutorial_jsma(train_start=0, train_end=60000, test_start=0,
                        test_end=10000, viz_enabled=True, nb_epochs=6,
                        batch_size=128, source_samples=10,
                        learning_rate=0.001):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    x_train, y_train, x_test, y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    nb_filters = 64
    # Define TF model graph
    model = ModelBasicCNN('model1', nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = LossCrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2017, 8, 30])
    train(sess, loss, x, y, x_train, y_train, args=train_params,
          rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    assert x_test.shape[0] == test_end - test_start, x_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')

    # Keep track of success (adversarial example mis-classified)
    results = np.zeros(source_samples, dtype='i')

    # Rate of perturbed features for each test set example
    perturbations = np.zeros(source_samples, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None, 'targetted': False}

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, source_samples):
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
        sample = x_test[sample_ind:(sample_ind+1)]

        # We are finding adversarial in Untargetted format
        current_class = int(np.argmax(y_test[sample_ind]))

        # Target now is the true class
        target = current_class
        print('Generating adv. example untargetted for class ',
              model_argmax(sess, x, preds, sample))

        # This call runs the Jacobian-based saliency map approach
        one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
        one_hot_target[0, target] = 1
        jsma_params['y_target'] = one_hot_target
        adv_x = jsma.generate_np(sample, **jsma_params)

        # Check if success was achieved
        res = int(model_argmax(sess, x, preds, adv_x) != target)

        print('Adversarial image  class ',
              model_argmax(sess, x, preds, adv_x))
        # Computer number of modified features
        adv_x_reshape = adv_x.reshape(-1)
        test_in_reshape = x_test[sample_ind].reshape(-1)
        nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
        percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

        # Display the original and adversarial images side-by-side
        if viz_enabled:
            figure = pair_visual(
                np.reshape(sample, (img_rows, img_cols, nchannels)),
                np.reshape(adv_x, (img_rows, img_cols, nchannels)), figure)

        # Update the arrays for later analysis
        results[sample_ind] = res
        perturbations[sample_ind] = percent_perturb

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    succ_rate = float(np.sum(results)) / source_samples
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
    report.clean_train_adv_eval = 1. - succ_rate

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))
    np.savetxt('untarpert.txt',perturbations)

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.4f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()

    return report


def main(argv=None):
    mnist_tutorial_jsma(viz_enabled=FLAGS.viz_enabled,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        source_samples=FLAGS.source_samples,
                        learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('source_samples', 20, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
