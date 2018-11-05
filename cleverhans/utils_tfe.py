"""
Utility functions for writing tf eager code
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

from cleverhans.loss import LossCrossEntropy
from cleverhans.model import Model
from .utils import batch_indices, _ArgsWrapper, create_logger

_logger = create_logger("cleverhans.utils.tfe")


def train(model, X_train=None, Y_train=None, save=False,
          predictions_adv=None, evaluate=None,
          args=None, rng=None, var_list=None,
          attack=None, attack_args=None):
  """
  Train a TF Eager model
  :param model: cleverhans.model.Model
  :param X_train: numpy array with training inputs
  :param Y_train: numpy array with training outputs
  :param save: boolean controlling the save operation
  :param predictions_adv: if set with the adversarial example tensor,
                          will run adversarial training
  :param evaluate: function that is run after each training iteration
                   (typically to display the test/validation accuracy).
  :param args: dict or argparse `Namespace` object.
               Should contain `nb_epochs`, `learning_rate`,
               `batch_size`
               If save is True, should also contain 'train_dir'
               and 'filename'
  :param rng: Instance of numpy.random.RandomState
  :param var_list: List of variables to train.
  :param attack: Instance of the class cleverhans.attacks.attacks_eager
  :param attack_args: Parameters required for the attack.
  :return: True if model trained
  """
  assert isinstance(model, Model)
  args = _ArgsWrapper(args or {})
  if ((attack is None) != (attack_args is None)):
    raise ValueError("attack and attack_args must be "
                     "passed together.")
  if X_train is None or Y_train is None:
    raise ValueError("X_train argument and Y_train argument "
                     "must be supplied.")
  # Check that necessary arguments were given (see doc above)
  assert args.nb_epochs, "Number of epochs was not given in args dict"
  assert args.learning_rate, "Learning rate was not given in args dict"
  assert args.batch_size, "Batch size was not given in args dict"

  if save:
    assert args.train_dir, "Directory for save was not given in args dict"
    assert args.filename, "Filename for save was not given in args dict"

  if rng is None:
    rng = np.random.RandomState()

  # Optimizer
  tfe = tf.contrib.eager
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  batch_x = tfe.Variable(X_train[0:args.batch_size], dtype=tf.float32)
  batch_y = tfe.Variable(Y_train[0:args.batch_size], dtype=tf.float32)

  # One epoch of training.
  for epoch in xrange(args.nb_epochs):
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_train)

    # Indices to shuffle training set
    index_shuf = list(range(len(X_train)))
    rng.shuffle(index_shuf)

    prev = time.time()
    for batch in range(nb_batches):

      # Compute batch start and end indices
      start, end = batch_indices(
          batch, len(X_train), args.batch_size)

      # Perform one training step
      tf.assign(batch_x, X_train[index_shuf[start:end]])
      tf.assign(batch_y, Y_train[index_shuf[start:end]])
      # Compute grads
      with tf.GradientTape() as tape:
        # Define loss
        loss_clean_obj = LossCrossEntropy(model, smoothing=0.)
        loss_clean = loss_clean_obj.fprop(x=batch_x, y=batch_y)
        loss = loss_clean
        # Adversarial training
        if attack is not None:
          batch_adv_x = attack.generate(batch_x, **attack_args)
          loss_adv_obj = LossCrossEntropy(model, smoothing=0.)
          loss_adv = loss_adv_obj.fprop(x=batch_adv_x, y=batch_y)
          loss = (loss_clean + loss_adv) / 2.0
      # Apply grads
      model_variables = model.get_params()
      grads = tape.gradient(loss, model_variables)
      optimizer.apply_gradients(zip(grads, model_variables))

    assert end >= len(X_train)  # Check that all examples were used
    cur = time.time()
    _logger.info("Epoch " + str(epoch) + " took " +
                 str(cur - prev) + " seconds")
    if evaluate is not None:
      evaluate()

  if save:
    save_path = os.path.join(args.train_dir, args.filename)
    saver = tf.train.Saver()
    saver.save(save_path, model_variables)
    _logger.info("Completed model training and saved at: " +
                 str(save_path))
  else:
    _logger.info("Completed model training.")

  return True


def model_eval(model, X_test=None, Y_test=None, args=None,
               attack=None, attack_args=None):
  """
  Compute the accuracy of a TF Eager model on some data
  :param model: instance of cleverhans.model.Model_Eager
                  with pretrained weights for evaluation.
  :param X_test: numpy array with training inputs
  :param Y_test: numpy array with training outputs
  :param args: dict or argparse `Namespace` object.
               Should contain `batch_size`
  :param attack: instance of the class cleverhans.attacks.attacks_eager
  :param attack_args: parameters required for the attack.
  :return: a float with the accuracy value
  """
  args = _ArgsWrapper(args or {})

  if ((attack is None) != (attack_args is None)):
    raise ValueError("attack and attack_args must be "
                     "passed together.")
  assert args.batch_size, "Batch size was not given in args dict"
  if X_test is None or Y_test is None:
    raise ValueError("X_test argument and Y_test argument "
                     "must be supplied.")

  # Init result var
  accuracy = 0.0

  # Compute number of batches
  nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
  assert nb_batches * args.batch_size >= len(X_test)

  X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                   dtype=X_test.dtype)
  Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                   dtype=Y_test.dtype)

  tfe = tf.contrib.eager
  batch_x = tfe.Variable(X_test[0:args.batch_size], dtype=tf.float32)
  batch_y = tfe.Variable(Y_test[0:args.batch_size], dtype=tf.float32)
  for batch in range(nb_batches):
    if batch % 100 == 0 and batch > 0:
      _logger.debug("Batch " + str(batch))

    # Must not use the `batch_indices` function here, because it
    # repeats some examples.
    # It's acceptable to repeat during training, but not eval.
    start = batch * args.batch_size
    end = min(len(X_test), start + args.batch_size)

    # The last batch may be smaller than all others. This should not
    # affect the accuarcy disproportionately.
    cur_batch_size = end - start
    X_cur[:cur_batch_size] = X_test[start:end]
    Y_cur[:cur_batch_size] = Y_test[start:end]
    tf.assign(batch_x, X_cur)
    tf.assign(batch_y, Y_cur)
    if attack is not None:
      batch_adv_x = attack.generate(batch_x, **attack_args)
      predictions = model.get_probs(batch_adv_x)
    else:
      predictions = model.get_probs(batch_x)
    cur_corr_preds = tf.equal(tf.argmax(batch_y, axis=-1),
                              tf.argmax(predictions, axis=-1))

    accuracy += cur_corr_preds.numpy()[:cur_batch_size].sum()

  assert end >= len(X_test)

  # Divide by number of examples to get final value
  accuracy /= len(X_test)

  return accuracy


def model_argmax(model, samples):
  """
  Helper function that computes the current class prediction
  :param samples: numpy array with input samples (dims must match x)
  :return: the argmax output of predictions, i.e. the current predicted class
  """
  tfe = tf.contrib.eager
  tf_samples = tfe.Variable(samples)
  probabilities = model.get_probs(tf_samples)

  if samples.shape[0] == 1:
    return tf.argmax(probabilities)
  else:
    return tf.argmax(probabilities, axis=1)
