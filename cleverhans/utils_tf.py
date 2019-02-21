"""Utility functions for writing TensorFlow code"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import math
import os
import time
import warnings

import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

from cleverhans.compat import device_lib
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger

_logger = create_logger("cleverhans.utils.tf")
_logger.setLevel(logging.INFO)


def model_loss(y, model, mean=True):
  """
  Define loss of TF graph
  :param y: correct labels
  :param model: output of the model
  :param mean: boolean indicating whether should return mean of loss
               or vector of losses for each input of the batch
  :return: return mean of loss if True, otherwise return vector with per
           sample loss
  """
  warnings.warn("This function is deprecated and will be removed on or after"
                " 2019-04-05. Switch to cleverhans.train.train.")
  op = model.op
  if op.type == "Softmax":
    logits, = op.inputs
  else:
    logits = model

  out = softmax_cross_entropy_with_logits(logits=logits, labels=y)

  if mean:
    out = reduce_mean(out)
  return out


def initialize_uninitialized_global_variables(sess):
  """
  Only initializes the variables of a TensorFlow session that were not
  already initialized.
  :param sess: the TensorFlow session
  :return:
  """
  # List all global variables
  global_vars = tf.global_variables()

  # Find initialized status for all variables
  is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
  is_initialized = sess.run(is_var_init)

  # List all variables that were not initialized previously
  not_initialized_vars = [var for (var, init) in
                          zip(global_vars, is_initialized) if not init]

  # Initialize all uninitialized variables found, if any
  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))


def train(sess, loss, x, y, X_train, Y_train, save=False,
          init_all=False, evaluate=None, feed=None, args=None,
          rng=None, var_list=None, fprop_args=None, optimizer=None):
  """
  Train a TF graph.
  This function is deprecated. Prefer cleverhans.train.train when possible.
  cleverhans.train.train supports multiple GPUs but this function is still
  needed to support legacy models that do not support calling fprop more
  than once.

  :param sess: TF session to use when training the graph
  :param loss: tensor, the model training loss.
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param X_train: numpy array with training inputs
  :param Y_train: numpy array with training outputs
  :param save: boolean controlling the save operation
  :param init_all: (boolean) If set to true, all TF variables in the session
                   are (re)initialized, otherwise only previously
                   uninitialized variables are initialized before training.
  :param evaluate: function that is run after each training iteration
                   (typically to display the test/validation accuracy).
  :param feed: An optional dictionary that is appended to the feeding
               dictionary before the session runs. Can be used to feed
               the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `nb_epochs`, `learning_rate`,
               `batch_size`
               If save is True, should also contain 'train_dir'
               and 'filename'
  :param rng: Instance of numpy.random.RandomState
  :param var_list: Optional list of parameters to train.
  :param fprop_args: dict, extra arguments to pass to fprop (loss and model).
  :param optimizer: Optimizer to be used for training
  :return: True if model trained
  """
  warnings.warn("This function is deprecated and will be removed on or after"
                " 2019-04-05. Switch to cleverhans.train.train.")

  args = _ArgsWrapper(args or {})
  fprop_args = fprop_args or {}

  # Check that necessary arguments were given (see doc above)
  assert args.nb_epochs, "Number of epochs was not given in args dict"
  if optimizer is None:
    assert args.learning_rate is not None, ("Learning rate was not given "
                                            "in args dict")
  assert args.batch_size, "Batch size was not given in args dict"

  if save:
    assert args.train_dir, "Directory for save was not given in args dict"
    assert args.filename, "Filename for save was not given in args dict"

  if rng is None:
    rng = np.random.RandomState()

  # Define optimizer
  loss_value = loss.fprop(x, y, **fprop_args)
  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  else:
    if not isinstance(optimizer, tf.train.Optimizer):
      raise ValueError("optimizer object must be from a child class of "
                       "tf.train.Optimizer")
  # Trigger update operations within the default graph (such as batch_norm).
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.minimize(loss_value, var_list=var_list)

  with sess.as_default():
    if hasattr(tf, "global_variables_initializer"):
      if init_all:
        tf.global_variables_initializer().run()
      else:
        initialize_uninitialized_global_variables(sess)
    else:
      warnings.warn("Update your copy of tensorflow; future versions of "
                    "CleverHans may drop support for this version.")
      sess.run(tf.initialize_all_variables())

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
        feed_dict = {x: X_train[index_shuf[start:end]],
                     y: Y_train[index_shuf[start:end]]}
        if feed is not None:
          feed_dict.update(feed)
        train_step.run(feed_dict=feed_dict)
      assert end >= len(X_train)  # Check that all examples were used
      cur = time.time()
      _logger.info("Epoch " + str(epoch) + " took " +
                   str(cur - prev) + " seconds")
      if evaluate is not None:
        evaluate()

    if save:
      save_path = os.path.join(args.train_dir, args.filename)
      saver = tf.train.Saver()
      saver.save(sess, save_path)
      _logger.info("Completed model training and saved at: " +
                   str(save_path))
    else:
      _logger.info("Completed model training.")

  return True


def model_eval(sess, x, y, predictions, X_test=None, Y_test=None,
               feed=None, args=None):
  """
  Compute the accuracy of a TF model on some data
  :param sess: TF session to use
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param predictions: model output predictions
  :param X_test: numpy array with training inputs
  :param Y_test: numpy array with training outputs
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `batch_size`
  :return: a float with the accuracy value
  """
  global _model_eval_cache
  args = _ArgsWrapper(args or {})

  assert args.batch_size, "Batch size was not given in args dict"
  if X_test is None or Y_test is None:
    raise ValueError("X_test argument and Y_test argument "
                     "must be supplied.")

  # Define accuracy symbolically
  key = (y, predictions)
  if key in _model_eval_cache:
    correct_preds = _model_eval_cache[key]
  else:
    correct_preds = tf.equal(tf.argmax(y, axis=-1),
                             tf.argmax(predictions, axis=-1))
    _model_eval_cache[key] = correct_preds

  # Init result var
  accuracy = 0.0

  with sess.as_default():
    # Compute number of batches
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)

    X_cur = np.zeros((args.batch_size,) + X_test.shape[1:],
                     dtype=X_test.dtype)
    Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:],
                     dtype=Y_test.dtype)
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
      feed_dict = {x: X_cur, y: Y_cur}
      if feed is not None:
        feed_dict.update(feed)
      cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

      accuracy += cur_corr_preds[:cur_batch_size].sum()

    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)

  return accuracy

_model_eval_cache = {}


def tf_model_load(sess, file_path=None):
  """

  :param sess: the session object to restore
  :param file_path: path to the restored session, if None is
                    taken from FLAGS.train_dir and FLAGS.filename
  :return:
  """
  with sess.as_default():
    saver = tf.train.Saver()
    if file_path is None:
      error = 'file_path argument is missing.'
      raise ValueError(error)
    saver.restore(sess, file_path)

  return True


def batch_eval(*args, **kwargs):
  """
  Wrapper around deprecated function.
  """
  # Inside function to avoid circular import
  from cleverhans.evaluation import batch_eval as new_batch_eval
  warnings.warn("batch_eval has moved to cleverhans.evaluation. "
                "batch_eval will be removed from utils_tf on or after "
                "2019-03-09.")
  return new_batch_eval(*args, **kwargs)


def model_argmax(sess, x, predictions, samples, feed=None):
  """
  Helper function that computes the current class prediction
  :param sess: TF session
  :param x: the input placeholder
  :param predictions: the model's symbolic output
  :param samples: numpy array with input samples (dims must match x)
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :return: the argmax output of predictions, i.e. the current predicted class
  """
  feed_dict = {x: samples}
  if feed is not None:
    feed_dict.update(feed)
  probabilities = sess.run(predictions, feed_dict)

  if samples.shape[0] == 1:
    return np.argmax(probabilities)
  else:
    return np.argmax(probabilities, axis=1)


def l2_batch_normalize(x, epsilon=1e-12, scope=None):
  """
  Helper function to normalize a batch of vectors.
  :param x: the input placeholder
  :param epsilon: stabilizes division
  :return: the batch of l2 normalized vector
  """
  with tf.name_scope(scope, "l2_batch_normalize") as name_scope:
    x_shape = tf.shape(x)
    x = tf.contrib.layers.flatten(x)
    x /= (epsilon + reduce_max(tf.abs(x), 1, keepdims=True))
    square_sum = reduce_sum(tf.square(x), 1, keepdims=True)
    x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
    x_norm = tf.multiply(x, x_inv_norm)
    return tf.reshape(x_norm, x_shape, name_scope)


def kl_with_logits(p_logits, q_logits, scope=None,
                   loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
  """Helper function to compute kl-divergence KL(p || q)
  """
  with tf.name_scope(scope, "kl_divergence") as name:
    p = tf.nn.softmax(p_logits)
    p_log = tf.nn.log_softmax(p_logits)
    q_log = tf.nn.log_softmax(q_logits)
    loss = reduce_mean(reduce_sum(p * (p_log - q_log), axis=1),
                       name=name)
    tf.losses.add_loss(loss, loss_collection)
    return loss


def clip_eta(eta, ord, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.ord norm ball
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')
  reduc_ind = list(xrange(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    eta = clip_by_value(eta, -eps, eps)
  else:
    if ord == 1:
      raise NotImplementedError("The expression below is not the correct way"
                                " to project onto the L1 norm ball.")
      norm = tf.maximum(avoid_zero_div,
                        reduce_sum(tf.abs(eta),
                                   reduc_ind, keepdims=True))
    elif ord == 2:
      # avoid_zero_div must go inside sqrt to avoid a divide by zero
      # in the gradient through this operation
      norm = tf.sqrt(tf.maximum(avoid_zero_div,
                                reduce_sum(tf.square(eta),
                                           reduc_ind,
                                           keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the
    # surface of the ball
    factor = tf.minimum(1., div(eps, norm))
    eta = eta * factor
  return eta


def model_train(sess, x, y, predictions, X_train, Y_train, save=False,
                predictions_adv=None, init_all=True, evaluate=None,
                feed=None, args=None, rng=None, var_list=None):
  """
  Train a TF graph
  :param sess: TF session to use when training the graph
  :param x: input placeholder
  :param y: output placeholder (for labels)
  :param predictions: model output predictions
  :param X_train: numpy array with training inputs
  :param Y_train: numpy array with training outputs
  :param save: boolean controlling the save operation
  :param predictions_adv: if set with the adversarial example tensor,
                          will run adversarial training
  :param init_all: (boolean) If set to true, all TF variables in the session
                   are (re)initialized, otherwise only previously
                   uninitialized variables are initialized before training.
  :param evaluate: function that is run after each training iteration
                   (typically to display the test/validation accuracy).
  :param feed: An optional dictionary that is appended to the feeding
               dictionary before the session runs. Can be used to feed
               the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
               Should contain `nb_epochs`, `learning_rate`,
               `batch_size`
               If save is True, should also contain 'train_dir'
               and 'filename'
  :param rng: Instance of numpy.random.RandomState
  :param var_list: Optional list of parameters to train.
  :return: True if model trained
  """
  warnings.warn("This function is deprecated and will be removed on or after"
                " 2019-04-05. Switch to cleverhans.train.train.")
  args = _ArgsWrapper(args or {})

  # Check that necessary arguments were given (see doc above)
  assert args.nb_epochs, "Number of epochs was not given in args dict"
  assert args.learning_rate, "Learning rate was not given in args dict"
  assert args.batch_size, "Batch size was not given in args dict"

  if save:
    assert args.train_dir, "Directory for save was not given in args dict"
    assert args.filename, "Filename for save was not given in args dict"

  if rng is None:
    rng = np.random.RandomState()

  # Define loss
  loss = model_loss(y, predictions)
  if predictions_adv is not None:
    loss = (loss + model_loss(y, predictions_adv)) / 2

  train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_step = train_step.minimize(loss, var_list=var_list)

  with sess.as_default():
    if hasattr(tf, "global_variables_initializer"):
      if init_all:
        tf.global_variables_initializer().run()
      else:
        initialize_uninitialized_global_variables(sess)
    else:
      warnings.warn("Update your copy of tensorflow; future versions of "
                    "CleverHans may drop support for this version.")
      sess.run(tf.initialize_all_variables())

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
        feed_dict = {x: X_train[index_shuf[start:end]],
                     y: Y_train[index_shuf[start:end]]}
        if feed is not None:
          feed_dict.update(feed)
        train_step.run(feed_dict=feed_dict)
      assert end >= len(X_train)  # Check that all examples were used
      cur = time.time()
      _logger.info("Epoch " + str(epoch) + " took " +
                   str(cur - prev) + " seconds")
      if evaluate is not None:
        evaluate()

    if save:
      save_path = os.path.join(args.train_dir, args.filename)
      saver = tf.train.Saver()
      saver.save(sess, save_path)
      _logger.info("Completed model training and saved at: " +
                   str(save_path))
    else:
      _logger.info("Completed model training.")

  return True


def infer_devices(devices=None):
  """
  Returns the list of devices that multi-replica code should use.
  :param devices: list of string device names, e.g. ["/GPU:0"]
      If the user specifies this, `infer_devices` checks that it is
      valid, and then uses this user-specified list.
      If the user does not specify this, infer_devices uses:
          - All available GPUs, if there are any
          - CPU otherwise
  """
  if devices is None:
    devices = get_available_gpus()
    if len(devices) == 0:
      warnings.warn("No GPUS, running on CPU")
      # Set device to empy string, tf will figure out whether to use
      # XLA or not, etc., automatically
      devices = [""]
  else:
    assert len(devices) > 0
    for device in devices:
      assert isinstance(device, six.string_types), type(device)
  return devices


def get_available_gpus():
  """
  Returns a list of string names of all available GPUs
  """
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def silence():
  """
  Silences tensorflaw's default printed messages
  """
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def clip_by_value(t, clip_value_min, clip_value_max, name=None):
  """
  A wrapper for clip_by_value that casts the clipping range if needed.
  """
  def cast_clip(clip):
    if t.dtype in (tf.float32, tf.float64):
      if hasattr(clip, 'dtype'):
        # Convert to tf dtype in case this is a numpy dtype
        clip_dtype = tf.as_dtype(clip.dtype)
        if clip_dtype != t.dtype:
          return tf.cast(clip, t.dtype)
    return clip

  clip_value_min = cast_clip(clip_value_min)
  clip_value_max = cast_clip(clip_value_max)

  return tf.clip_by_value(t, clip_value_min, clip_value_max, name)

def mul(a, b):
  """
  A wrapper around tf multiplication that does more automatic casting of
  the input.
  """
  def multiply(a, b):
    return a * b
  return op_with_scalar_cast(a, b, multiply)

def div(a, b):
  """
  A wrapper around tf division that does more automatic casting of
  the input.
  """
  def divide(a, b):
    return a / b
  return op_with_scalar_cast(a, b, divide)

def op_with_scalar_cast(a, b, f):
  """
  Builds the graph to compute f(a, b).
  If only one of the two arguments is a scalar and the operation would
  cause a type error without casting, casts the scalar to match the
  tensor.
  :param a: a tf-compatible array or scalar
  :param b: a tf-compatible array or scalar
  """

  try:
    return f(a, b)
  except (TypeError, ValueError):
    pass

  def is_scalar(x):
    if hasattr(x, "get_shape"):
      shape = x.get_shape()
      return shape.ndims == 0
    if hasattr(x, "ndim"):
      return x.ndim == 0
    assert isinstance(x, (int, float))
    return True

  a_scalar = is_scalar(a)
  b_scalar = is_scalar(b)

  if a_scalar and b_scalar:
    raise TypeError("Trying to apply " + str(f) + " with mixed types")

  if a_scalar and not b_scalar:
    a = tf.cast(a, b.dtype)

  if b_scalar and not a_scalar:
    b = tf.cast(b, a.dtype)

  return f(a, b)

def assert_less_equal(*args, **kwargs):
  """
  Wrapper for tf.assert_less_equal
  Overrides tf.device so that the assert always goes on CPU.
  The unwrapped version raises an exception if used with tf.device("/GPU:x").
  """
  with tf.device("/CPU:0"):
    return tf.assert_less_equal(*args, **kwargs)

def assert_greater_equal(*args, **kwargs):
  """
  Wrapper for tf.assert_greater_equal.
  Overrides tf.device so that the assert always goes on CPU.
  The unwrapped version raises an exception if used with tf.device("/GPU:x").
  """
  with tf.device("/CPU:0"):
    return tf.assert_greater_equal(*args, **kwargs)

def assert_equal(*args, **kwargs):
  """
  Wrapper for tf.assert_equal.
  Overrides tf.device so that the assert always goes on CPU.
  The unwrapped version raises an exception if used with tf.device("/GPU:x").
  """
  with tf.device("/CPU:0"):
    return tf.assert_equal(*args, **kwargs)

def jacobian_graph(predictions, x, nb_classes):
  """
  Create the Jacobian graph to be ran later in a TF session
  :param predictions: the model's symbolic output (linear output,
      pre-softmax)
  :param x: the input placeholder
  :param nb_classes: the number of classes the model has
  :return:
  """

  # This function will return a list of TF gradients
  list_derivatives = []

  # Define the TF graph elements to compute our derivatives for each class
  for class_ind in xrange(nb_classes):
    derivatives, = tf.gradients(predictions[:, class_ind], x)
    list_derivatives.append(derivatives)

  return list_derivatives

def jacobian_augmentation(sess,
                          x,
                          X_sub_prev,
                          Y_sub,
                          grads,
                          lmbda,
                          aug_batch_size=512,
                          feed=None):
  """
  Augment an adversary's substitute training set using the Jacobian
  of a substitute model to generate new synthetic inputs.
  See https://arxiv.org/abs/1602.02697 for more details.
  See cleverhans_tutorials/mnist_blackbox.py for example use case
  :param sess: TF session in which the substitute model is defined
  :param x: input TF placeholder for the substitute model
  :param X_sub_prev: substitute training data available to the adversary
                     at the previous iteration
  :param Y_sub: substitute training labels available to the adversary
                at the previous iteration
  :param grads: Jacobian symbolic graph for the substitute
                (should be generated using utils_tf.jacobian_graph)
  :return: augmented substitute data (will need to be labeled by oracle)
  """
  assert len(x.get_shape()) == len(np.shape(X_sub_prev))
  assert len(grads) >= np.max(Y_sub) + 1
  assert len(X_sub_prev) == len(Y_sub)

  aug_batch_size = min(aug_batch_size, X_sub_prev.shape[0])

  # Prepare input_shape (outside loop) for feeding dictionary below
  input_shape = list(x.get_shape())
  input_shape[0] = 1

  # Create new numpy array for adversary training data
  # with twice as many components on the first dimension.
  X_sub = np.vstack([X_sub_prev, X_sub_prev])
  num_samples = X_sub_prev.shape[0]

  # Creating and processing as batch
  for p_idxs in range(0, num_samples, aug_batch_size):
    X_batch = X_sub_prev[p_idxs:p_idxs + aug_batch_size, ...]
    feed_dict = {x: X_batch}
    if feed is not None:
      feed_dict.update(feed)

    # Compute sign matrix
    grad_val = sess.run([tf.sign(grads)], feed_dict=feed_dict)[0]

    # Create new synthetic point in adversary substitute training set
    for (indx, ind) in zip(range(p_idxs, p_idxs + X_batch.shape[0]),
                           range(X_batch.shape[0])):
      X_sub[num_samples + indx] = (
          X_batch[ind] + lmbda * grad_val[Y_sub[indx], ind, ...])

  # Return augmented training data (needs to be labeled afterwards)
  return X_sub
