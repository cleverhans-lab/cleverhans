"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow Eager.

It is similar to mnist_tutorial_tf.py.
mnist_tutorial_tf - dynaminc eager execution.
mnist_tutorial_tf - graph based execution.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf

from cleverhans.compat import flags
from cleverhans.utils import AccuracyReport
from cleverhans.utils_tfe import train, model_eval
from cleverhans.attacks_tfe import BasicIterativeMethod
from cleverhans.attacks_tfe import FastGradientMethod
from cleverhans.dataset import MNIST
from cleverhans.utils import set_log_level
from cleverhans_tutorials.tutorial_models_tfe import ModelBasicCNNTFE

if tf.executing_eagerly() is True:
  print('TF Eager Activated.')
else:
  raise Exception("Error Enabling Eager Execution.")
tfe = tf.contrib.eager

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
NB_FILTERS = 64

# Keeps track of implemented attacks.
# Maps attack string taken from bash to attack class
# -- 'fgsm' : FastGradientMethod
# -- 'bim': BasicIterativeMethod

AVAILABLE_ATTACKS = {
    'fgsm': FastGradientMethod,
    'bim': BasicIterativeMethod
}


def attack_selection(attack_string):
  """
  Selects the Attack Class using string input.
  :param attack_string: adversarial attack name in string format
  :return: attack class defined in cleverhans.attacks_eager
  """

  # List of Implemented attacks
  attacks_list = AVAILABLE_ATTACKS.keys()

  # Checking for  requested attack in list of available attacks.
  if attack_string is None:
    raise AttributeError("Attack type is not specified, "
                         "list of available attacks\t".join(attacks_list))
  if attack_string not in attacks_list:
    raise AttributeError("Attack not available "
                         "list of available attacks\t".join(attacks_list))
  # Mapping attack from string to class.
  attack_class = AVAILABLE_ATTACKS[attack_string]
  return attack_class


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=NB_FILTERS, num_threads=None,
                   attack_string=None):
  """
  MNIST cleverhans tutorial
  :param train_start: index of first training set example.
  :param train_end: index of last training set example.
  :param test_start: index of first test set example.
  :param test_end: index of last test set example.
  :param nb_epochs: number of epochs to train model.
  :param batch_size: size of training batches.
  :param learning_rate: learning rate for training.
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
  :param testing: if true, complete an AccuracyReport for unit tests
                  to verify that performance is adequate.
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param nb_filters: number of filters in the CNN used for training.
  :param num_threads: number of threads used for running the process.
  :param attack_string: attack name for crafting adversarial attacks and
                          adversarial training, in string format.
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Get MNIST test data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  X_train, Y_train = mnist.get_set('train')
  X_test, Y_test = mnist.get_set('test')

  # Use label smoothing
  assert Y_train.shape[1] == 10
  label_smooth = .1
  Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }

  # Initialize the attack object
  attack_class = attack_selection(attack_string)
  attack_params = {'eps': 0.3, 'clip_min': 0.,
                   'clip_max': 1.}

  rng = np.random.RandomState([2018, 6, 18])
  if clean_train:
    model = ModelBasicCNNTFE(nb_filters=nb_filters)

    def evaluate_clean():
      """Evaluate the accuracy of the MNIST model on legitimate test
      examples
      """
      eval_params = {'batch_size': batch_size}
      acc = model_eval(model, X_test, Y_test, args=eval_params)
      report.clean_train_clean_eval = acc
      assert X_test.shape[0] == test_end - test_start, X_test.shape
      print('Test accuracy on legitimate examples: %0.4f' % acc)

    train(model, X_train, Y_train, evaluate=evaluate_clean,
          args=train_params, rng=rng, var_list=model.get_params())

    if testing:
      # Calculate training error
      eval_params = {'batch_size': batch_size}
      acc = model_eval(model, X_train, Y_train, args=eval_params)
      report.train_clean_train_clean_eval = acc

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    attack = attack_class(model)
    acc = model_eval(
        model, X_test, Y_test, args=eval_par,
        attack=attack, attack_args=attack_params)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    # Calculate training error
    if testing:
      eval_par = {'batch_size': batch_size}
      acc = model_eval(
          model, X_train, Y_train, args=eval_par,
          attack=attack, attack_args=attack_params)
      print('Train accuracy on adversarial examples: %0.4f\n' % acc)
      report.train_clean_train_adv_eval = acc

    attack = None
    print("Repeating the process, using adversarial training")

  model_adv_train = ModelBasicCNNTFE(nb_filters=nb_filters)
  attack = attack_class(model_adv_train)

  def evaluate_adv():
    # Accuracy of adversarially trained model on legitimate test inputs
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(
        model_adv_train, X_test, Y_test,
        args=eval_params)
    print('Test accuracy on legitimate examples: %0.4f' % accuracy)
    report.adv_train_clean_eval = accuracy
    # Accuracy of the adversarially trained model on adversarial examples
    accuracy = model_eval(
        model_adv_train, X_test, Y_test,
        args=eval_params, attack=attack,
        attack_args=attack_params)
    print('Test accuracy on adversarial examples: %0.4f' % accuracy)
    report.adv_train_adv_eval = accuracy

  # Perform and evaluate adversarial training
  train(model_adv_train, X_train, Y_train, evaluate=evaluate_adv,
        args=train_params, rng=rng,
        var_list=model_adv_train.get_params(),
        attack=attack, attack_args=attack_params)

  # Calculate training errors
  if testing:
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(
        model_adv_train, X_train, Y_train, args=eval_params,
        attack=None, attack_args=None)
    report.train_adv_train_clean_eval = accuracy
    accuracy = model_eval(
        model_adv_train, X_train, Y_train, args=eval_params,
        attack=attack, attack_args=attack_params)
    report.train_adv_train_adv_eval = accuracy
  return report


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(
      nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate, clean_train=FLAGS.clean_train,
      backprop_through_attack=FLAGS.backprop_through_attack,
      nb_filters=FLAGS.nb_filters, attack_string=FLAGS.attack)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
  flags.DEFINE_bool('backprop_through_attack', False,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))
  flags.DEFINE_string('attack', 'fgsm',
                      'Adversarial attack crafted and used for training')
  tf.app.run()
