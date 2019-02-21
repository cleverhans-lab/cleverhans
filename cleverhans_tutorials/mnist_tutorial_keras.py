"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with Keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, testing=False,
                   label_smoothing=0.1):
  """
  MNIST CleverHans tutorial
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param testing: if true, training error is calculated
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)
  # Force TensorFlow to use single thread to improve reproducibility
  config = tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=1)

  if keras.backend.image_data_format() != 'channels_last':
    raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

  # Create TF session and set as Keras backend session
  sess = tf.Session(config=config)
  keras.backend.set_session(sess)

  # Get MNIST test data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Obtain Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Label smoothing
  y_train -= label_smoothing * (y_train - 1. / nb_classes)

  # Define Keras model
  model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels, nb_filters=64,
                    nb_classes=nb_classes)
  print("Defined Keras model.")

  # To be able to call the model in the custom loss, we need to call it once
  # before, see https://github.com/tensorflow/tensorflow/issues/23769
  model(model.input)

  # Initialize the Fast Gradient Sign Method (FGSM) attack object
  wrap = KerasModelWrapper(model)
  fgsm = FastGradientMethod(wrap, sess=sess)
  fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}

  adv_acc_metric = get_adversarial_acc_metric(model, fgsm, fgsm_params)
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy', adv_acc_metric]
  )

  # Train an MNIST model
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=nb_epochs,
            validation_data=(x_test, y_test),
            verbose=2)

  # Evaluate the accuracy on legitimate and adversarial test examples
  _, acc, adv_acc = model.evaluate(x_test, y_test,
                                   batch_size=batch_size,
                                   verbose=0)
  report.clean_train_clean_eval = acc
  report.clean_train_adv_eval = adv_acc
  print('Test accuracy on legitimate examples: %0.4f' % acc)
  print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

  # Calculate training error
  if testing:
    _, train_acc, train_adv_acc = model.evaluate(x_train, y_train,
                                                 batch_size=batch_size,
                                                 verbose=0)
    report.train_clean_train_clean_eval = train_acc
    report.train_clean_train_adv_eval = train_adv_acc

  print("Repeating the process, using adversarial training")
  # Redefine Keras model
  model_2 = cnn_model(img_rows=img_rows, img_cols=img_cols,
                      channels=nchannels, nb_filters=64,
                      nb_classes=nb_classes)
  model_2(model_2.input)
  wrap_2 = KerasModelWrapper(model_2)
  fgsm_2 = FastGradientMethod(wrap_2, sess=sess)

  # Use a loss function based on legitimate and adversarial examples
  adv_loss_2 = get_adversarial_loss(model_2, fgsm_2, fgsm_params)
  adv_acc_metric_2 = get_adversarial_acc_metric(model_2, fgsm_2, fgsm_params)
  model_2.compile(
      optimizer=keras.optimizers.Adam(learning_rate),
      loss=adv_loss_2,
      metrics=['accuracy', adv_acc_metric_2]
  )

  # Train an MNIST model
  model_2.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nb_epochs,
              validation_data=(x_test, y_test),
              verbose=2)

  # Evaluate the accuracy on legitimate and adversarial test examples
  _, acc, adv_acc = model_2.evaluate(x_test, y_test,
                                     batch_size=batch_size,
                                     verbose=0)
  report.adv_train_clean_eval = acc
  report.adv_train_adv_eval = adv_acc
  print('Test accuracy on legitimate examples: %0.4f' % acc)
  print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

  # Calculate training error
  if testing:
    _, train_acc, train_adv_acc = model_2.evaluate(x_train, y_train,
                                                   batch_size=batch_size,
                                                   verbose=0)
    report.train_adv_train_clean_eval = train_acc
    report.train_adv_train_adv_eval = train_adv_acc

  return report


def get_adversarial_acc_metric(model, fgsm, fgsm_params):
  def adv_acc(y, _):
    # Generate adversarial examples
    x_adv = fgsm.generate(model.input, **fgsm_params)
    # Consider the attack to be constant
    x_adv = tf.stop_gradient(x_adv)

    # Accuracy on the adversarial examples
    preds_adv = model(x_adv)
    return keras.metrics.categorical_accuracy(y, preds_adv)

  return adv_acc


def get_adversarial_loss(model, fgsm, fgsm_params):
  def adv_loss(y, preds):
    # Cross-entropy on the legitimate examples
    cross_ent = keras.losses.categorical_crossentropy(y, preds)

    # Generate adversarial examples
    x_adv = fgsm.generate(model.input, **fgsm_params)
    # Consider the attack to be constant
    x_adv = tf.stop_gradient(x_adv)

    # Cross-entropy on the adversarial examples
    preds_adv = model(x_adv)
    cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

    return 0.5 * cross_ent + 0.5 * cross_ent_adv

  return adv_loss


def main(argv=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  tf.app.run()
