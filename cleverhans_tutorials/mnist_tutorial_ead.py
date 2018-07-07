"""
This tutorial shows how to generate adversarial examples
using the EAD attack in white-box setting.
The original paper can be found at:
https://arxiv.org/pdf/1709.04114.pdf
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
import os
from cleverhans.attacks import ElasticNetMethod
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS


def mnist_tutorial_ead(train_start=0, train_end=60000, test_start=0,
                       test_end=10000, viz_enabled=True, nb_epochs=6,
                       batch_size=128, nb_classes=10, source_samples=10,
                       learning_rate=0.001, attack_iterations=1000,
                       model_path=os.path.join("models", "mnist"),
                       targeted=True):
    """
    MNIST tutorial for the EAD attack
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
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
        'filename': os.path.split(model_path)[-1]
    }

    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
    else:
        model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                    save=os.path.exists("models"), rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using the Elastic Net Method
    ###########################################################################
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a CW attack object
    ead = ElasticNetMethod(model, back='tf', sess=sess)

    if viz_enabled:
        assert source_samples == nb_classes
        idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][0]
                for i in range(nb_classes)]
    if targeted:
        if viz_enabled:
            # Initialize our array for grid visualization
            grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            adv_inputs = np.array(
                [[instance] * nb_classes for instance in X_test[idxs]],
                dtype=np.float32)
        else:
            adv_inputs = np.array(
                [[instance] * nb_classes for
                 instance in X_test[:source_samples]], dtype=np.float32)

        one_hot = np.zeros((nb_classes, nb_classes))
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

        adv_inputs = adv_inputs.reshape(
            (source_samples * nb_classes, img_rows, img_cols, 1))
        adv_ys = np.array([one_hot] * source_samples,
                          dtype=np.float32).reshape((source_samples *
                                                     nb_classes, nb_classes))
        yname = "y_target"
    else:
        if viz_enabled:
            # Initialize our array for grid visualization
            grid_shape = (nb_classes, 2, img_rows, img_cols, channels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            adv_inputs = X_test[idxs]
        else:
            adv_inputs = X_test[:source_samples]

        adv_ys = None
        yname = "y"
    """
    No bss for computational efficiency, due to this 'initial_const'
    fixed to high value. Success rate should be unchanged from default
    configuration, however, average distortion will be increased
    """
    ead_params = {'binary_search_steps': 1,
                  yname: adv_ys,
                  'max_iterations': attack_iterations,
                  'batch_size': source_samples * nb_classes if
                  targeted else source_samples,
                  'initial_const': 1e+5}

    adv = ead.generate_np(adv_inputs,
                          **ead_params)

    eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}
    if targeted:
        adv_accuracy = model_eval(
            sess, x, y, preds, adv, adv_ys, args=eval_params)
    else:
        if viz_enabled:
            adv_accuracy = 1 - \
                model_eval(sess, x, y, preds, adv, Y_test[
                           idxs], args=eval_params)
        else:
            adv_accuracy = 1 - \
                model_eval(sess, x, y, preds, adv, Y_test[
                           :source_samples], args=eval_params)

    if viz_enabled:
        for j in range(nb_classes):
            if targeted:
                for i in range(nb_classes):
                    grid_viz_data[i, j] = adv[i * nb_classes + j]
            else:
                grid_viz_data[j, 0] = adv_inputs[j]
                grid_viz_data[j, 1] = adv[j]

        print(grid_viz_data.shape)

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average L2 distortion introduced by the algorithm
    percent_perturbed_l2 = np.mean(np.sum((adv - adv_inputs)**2,
                                          axis=(1, 2, 3))**.5)
    print('Avg. L_2 perturbation norm {0:.4f}'.format(percent_perturbed_l2))

    # Compute the average L1 distortion introduced by the algorithm
    percent_perturbed_l1 = np.mean(np.sum(np.abs(adv - adv_inputs),
                                          axis=(1, 2, 3)))
    print('Avg. L_1 perturbation norm {0:.4f}'.format(percent_perturbed_l1))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
        import matplotlib.pyplot as plt
        _ = grid_visual(grid_viz_data)

    return report


def main(argv=None):
    mnist_tutorial_ead(viz_enabled=FLAGS.viz_enabled,
                       nb_epochs=FLAGS.nb_epochs,
                       batch_size=FLAGS.batch_size,
                       nb_classes=FLAGS.nb_classes,
                       source_samples=FLAGS.source_samples,
                       learning_rate=FLAGS.learning_rate,
                       attack_iterations=FLAGS.attack_iterations,
                       model_path=FLAGS.model_path,
                       targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', 1000,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', True,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
