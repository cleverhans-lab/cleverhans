"""
This tutorial shows how to generate adversarial examples
using FGSM in black-box setting.
The original paper can be found at:
https://arxiv.org/abs/1602.02697
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import jacobian_graph, jacobian_augmentation
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.initializers import HeReLuNormalInitializer
from cleverhans.loss import CrossEntropy
from cleverhans.model import Model
from cleverhans.train import train
from cleverhans.utils import set_log_level
from cleverhans.utils import TemporaryLogLevel
from cleverhans.utils import to_categorical
from cleverhans.utils_tf import model_eval, batch_eval

from cleverhans.model_zoo.basic_cnn import ModelBasicCNN

FLAGS = flags.FLAGS

NB_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = .001
NB_EPOCHS = 10
HOLDOUT = 150
DATA_AUG = 6
NB_EPOCHS_S = 10
LMBDA = .1
AUG_BATCH_SIZE = 512


def setup_tutorial():
  """
  Helper function to check correct configuration of tf for tutorial
  :return: True if setup checks completed
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  return True


def prep_bbox(sess, x, y, x_train, y_train, x_test, y_test,
              nb_epochs, batch_size, learning_rate,
              rng, nb_classes=10, img_rows=28, img_cols=28, nchannels=1):
  """
  Define and train a model that simulates the "remote"
  black-box oracle described in the original paper.
  :param sess: the TF session
  :param x: the input placeholder for MNIST
  :param y: the ouput placeholder for MNIST
  :param x_train: the training data for the oracle
  :param y_train: the training labels for the oracle
  :param x_test: the testing data for the oracle
  :param y_test: the testing labels for the oracle
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param rng: numpy.random.RandomState
  :return:
  """

  # Define TF model graph (for the black-box model)
  nb_filters = 64
  model = ModelBasicCNN('model1', nb_classes, nb_filters)
  loss = CrossEntropy(model, smoothing=0.1)
  predictions = model.get_logits(x)
  print("Defined TensorFlow model graph.")

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  train(sess, loss, x_train, y_train, args=train_params, rng=rng)

  # Print out the accuracy on legitimate data
  eval_params = {'batch_size': batch_size}
  accuracy = model_eval(sess, x, y, predictions, x_test, y_test,
                        args=eval_params)
  print('Test accuracy of black-box on legitimate test '
        'examples: ' + str(accuracy))

  return model, predictions, accuracy


class ModelSubstitute(Model):
  def __init__(self, scope, nb_classes, nb_filters=200, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = nb_filters

  def fprop(self, x, **kwargs):
    del kwargs
    my_dense = functools.partial(
        tf.layers.dense, kernel_initializer=HeReLuNormalInitializer)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      y = tf.layers.flatten(x)
      y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
      y = my_dense(y, self.nb_filters, activation=tf.nn.relu)
      logits = my_dense(y, self.nb_classes)
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}


def train_sub(sess, x, y, bbox_preds, x_sub, y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              aug_batch_size, rng, img_rows=28, img_cols=28,
              nchannels=1):
  """
  This function creates the substitute by alternatively
  augmenting the training data and training the substitute.
  :param sess: TF session
  :param x: input TF placeholder
  :param y: output TF placeholder
  :param bbox_preds: output of black-box model predictions
  :param x_sub: initial substitute training data
  :param y_sub: initial substitute training labels
  :param nb_classes: number of output classes
  :param nb_epochs_s: number of epochs to train substitute model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param data_aug: number of times substitute training data is augmented
  :param lmbda: lambda from arxiv.org/abs/1602.02697
  :param rng: numpy.random.RandomState instance
  :return:
  """
  # Define TF model graph (for the black-box model)
  model_sub = ModelSubstitute('model_s', nb_classes)
  preds_sub = model_sub.get_logits(x)
  loss_sub = CrossEntropy(model_sub, smoothing=0)

  print("Defined TensorFlow model graph for the substitute.")

  # Define the Jacobian symbolically using TensorFlow
  grads = jacobian_graph(preds_sub, x, nb_classes)

  # Train the substitute and augment dataset alternatively
  for rho in xrange(data_aug):
    print("Substitute training epoch #" + str(rho))
    train_params = {
        'nb_epochs': nb_epochs_s,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
      train(sess, loss_sub, x_sub, to_categorical(y_sub, nb_classes),
            init_all=False, args=train_params, rng=rng,
            var_list=model_sub.get_params())

    # If we are not at last substitute training iteration, augment dataset
    if rho < data_aug - 1:
      print("Augmenting substitute training data.")
      # Perform the Jacobian augmentation
      lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
      x_sub = jacobian_augmentation(sess, x, x_sub, y_sub, grads,
                                    lmbda_coef * lmbda, aug_batch_size)

      print("Labeling substitute training data.")
      # Label the newly generated synthetic points using the black-box
      y_sub = np.hstack([y_sub, y_sub])
      x_sub_prev = x_sub[int(len(x_sub)/2):]
      eval_params = {'batch_size': batch_size}
      bbox_val = batch_eval(sess, [x], [bbox_preds], [x_sub_prev],
                            args=eval_params)[0]
      # Note here that we take the argmax because the adversary
      # only has access to the label (not the probabilities) output
      # by the black-box model
      y_sub[int(len(x_sub)/2):] = np.argmax(bbox_val, axis=1)

  return model_sub, preds_sub


def mnist_blackbox(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=NB_CLASSES,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                   nb_epochs=NB_EPOCHS, holdout=HOLDOUT, data_aug=DATA_AUG,
                   nb_epochs_s=NB_EPOCHS_S, lmbda=LMBDA,
                   aug_batch_size=AUG_BATCH_SIZE):
  """
  MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :return: a dictionary with:
           * black-box model accuracy on test set
           * substitute model accuracy on test set
           * black-box model accuracy on adversarial examples transferred
             from the substitute model
  """

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Dictionary used to keep track and return key accuracies
  accuracies = {}

  # Perform tutorial setup
  assert setup_tutorial()

  # Create TF session
  sess = tf.Session()

  # Get MNIST data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Initialize substitute training set reserved for adversary
  x_sub = x_test[:holdout]
  y_sub = np.argmax(y_test[:holdout], axis=1)

  # Redefine test set as remaining samples unavailable to adversaries
  x_test = x_test[holdout:]
  y_test = y_test[holdout:]

  # Obtain Image parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Seed random number generator so tutorial is reproducible
  rng = np.random.RandomState([2017, 8, 30])

  # Simulate the black-box model locally
  # You could replace this by a remote labeling API for instance
  print("Preparing the black-box model.")
  prep_bbox_out = prep_bbox(sess, x, y, x_train, y_train, x_test, y_test,
                            nb_epochs, batch_size, learning_rate,
                            rng, nb_classes, img_rows, img_cols, nchannels)
  model, bbox_preds, accuracies['bbox'] = prep_bbox_out

  # Train substitute using method from https://arxiv.org/abs/1602.02697
  print("Training the substitute model.")
  train_sub_out = train_sub(sess, x, y, bbox_preds, x_sub, y_sub,
                            nb_classes, nb_epochs_s, batch_size,
                            learning_rate, data_aug, lmbda, aug_batch_size,
                            rng, img_rows, img_cols, nchannels)
  model_sub, preds_sub = train_sub_out

  # Evaluate the substitute model on clean test examples
  eval_params = {'batch_size': batch_size}
  acc = model_eval(sess, x, y, preds_sub, x_test, y_test, args=eval_params)
  accuracies['sub'] = acc

  # Initialize the Fast Gradient Sign Method (FGSM) attack object.
  fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
  fgsm = FastGradientMethod(model_sub, sess=sess)

  # Craft adversarial examples using the substitute
  eval_params = {'batch_size': batch_size}
  x_adv_sub = fgsm.generate(x, **fgsm_par)

  # Evaluate the accuracy of the "black-box" model on adversarial examples
  accuracy = model_eval(sess, x, y, model.get_logits(x_adv_sub),
                        x_test, y_test, args=eval_params)
  print('Test accuracy of oracle on adversarial examples generated '
        'using the substitute: ' + str(accuracy))
  accuracies['bbox_on_sub_adv_ex'] = accuracy

  return accuracies


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_blackbox(nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                 data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                 lmbda=FLAGS.lmbda, aug_batch_size=FLAGS.data_aug_batch_size)


if __name__ == '__main__':

  # General flags
  flags.DEFINE_integer('nb_classes', NB_CLASSES,
                       'Number of classes in problem')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  # Flags related to oracle
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')

  # Flags related to substitute
  flags.DEFINE_integer('holdout', HOLDOUT,
                       'Test set holdout for adversary')
  flags.DEFINE_integer('data_aug', DATA_AUG,
                       'Number of substitute data augmentations')
  flags.DEFINE_integer('nb_epochs_s', NB_EPOCHS_S,
                       'Training epochs for substitute')
  flags.DEFINE_float('lmbda', LMBDA, 'Lambda from arxiv.org/abs/1602.02697')
  flags.DEFINE_integer('data_aug_batch_size', AUG_BATCH_SIZE,
                       'Batch size for augmentation')

  tf.app.run()
