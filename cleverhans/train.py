"""
Multi-replica synchronous training


NOTE: This module is much more free to change than many other modules
in CleverHans. CleverHans is very conservative about changes to any
code that affects the output of benchmark tests (attacks, evaluation
methods, etc.). This module provides *model training* functionality
not *benchmarks* and thus is free to change rapidly to provide better
speed, accuracy, etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.utils import _ArgsWrapper, create_logger
from cleverhans.utils import safe_zip
from cleverhans.utils_tf import infer_devices
from cleverhans.utils_tf import initialize_uninitialized_global_variables


_logger = create_logger("train")
_logger.setLevel(logging.INFO)


def train(sess, loss, x_train, y_train,
          init_all=True, evaluate=None, feed=None, args=None,
          rng=None, var_list=None, fprop_args=None, optimizer=None,
          devices=None, x_batch_preprocessor=None, use_ema=False,
          ema_decay=.998, run_canary=True,
          loss_threshold=1e5):
  """
  Run (optionally multi-replica, synchronous) training to minimize `loss`
  :param sess: TF session to use when training the graph
  :param loss: tensor, the loss to minimize
  :param x_train: numpy array with training inputs
  :param y_train: numpy array with training outputs
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
  :param rng: Instance of numpy.random.RandomState
  :param var_list: Optional list of parameters to train.
  :param fprop_args: dict, extra arguments to pass to fprop (loss and model).
  :param optimizer: Optimizer to be used for training
  :param devices: list of device names to use for training
      If None, defaults to: all GPUs, if GPUs are available
                            all devices, if no GPUs are available
  :param x_batch_preprocessor: callable
      Takes a single tensor containing an x_train batch as input
      Returns a single tensor containing an x_train batch as output
      Called to preprocess the data before passing the data to the Loss
  :param use_ema: bool
      If true, uses an exponential moving average of the model parameters
  :param ema_decay: float or callable
      The decay parameter for EMA, if EMA is used
      If a callable rather than a float, this is a callable that takes
      the epoch and batch as arguments and returns the ema_decay for
      the current batch.
  :param run_canary: bool
      If True and using 3 or more GPUs, runs some canary code that should
      fail if there is a multi-GPU driver problem.
      Turn this off if your gradients are inherently stochastic (e.g.
      if you use dropout). The canary code checks that all GPUs give
      approximately the same gradient.
  :param loss_threshold: float
      Raise an exception if the loss exceeds this value.
      This is intended to rapidly detect numerical problems.
      Sometimes the loss may legitimately be higher than this value. In
      such cases, raise the value. If needed it can be np.inf.
  :return: True if model trained
  """
  args = _ArgsWrapper(args or {})
  fprop_args = fprop_args or {}

  # Check that necessary arguments were given (see doc above)
  assert args.nb_epochs, "Number of epochs was not given in args dict"
  if optimizer is None:
    if args.learning_rate is None:
      raise ValueError("Learning rate was not given in args dict")
  assert args.batch_size, "Batch size was not given in args dict"

  if rng is None:
    rng = np.random.RandomState()

  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  else:
    if not isinstance(optimizer, tf.train.Optimizer):
      raise ValueError("optimizer object must be from a child class of "
                       "tf.train.Optimizer")

  grads = []
  xs = []
  preprocessed_xs = []
  ys = []

  devices = infer_devices(devices)
  for device in devices:
    with tf.device(device):
      x = tf.placeholder(x_train.dtype, (None,) + x_train.shape[1:])
      y = tf.placeholder(x_train.dtype, (None,) + y_train.shape[1:])
      xs.append(x)
      ys.append(y)

      if x_batch_preprocessor is not None:
        x = x_batch_preprocessor(x)

      # We need to keep track of these so that the canary can feed
      # preprocessed values. If the canary had to feed raw values,
      # stochastic preprocessing could make the canary fail.
      preprocessed_xs.append(x)

      loss_value = loss.fprop(x, y, **fprop_args)

      grads.append(optimizer.compute_gradients(
          loss_value, var_list=var_list))
  num_devices = len(devices)
  print("num_devices: ", num_devices)

  grad = avg_grads(grads)
  # Trigger update operations within the default graph (such as batch_norm).
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.apply_gradients(grad)

  epoch_tf = tf.placeholder(tf.int32, [])
  batch_tf = tf.placeholder(tf.int32, [])

  if use_ema:
    if callable(ema_decay):
      ema_decay = ema_decay(epoch_tf, batch_tf)
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    with tf.control_dependencies([train_step]):
      train_step = ema.apply(var_list)
    # Get pointers to the EMA's running average variables
    avg_params = [ema.average(param) for param in var_list]
    # Make temporary buffers used for swapping the live and running average
    # parameters
    tmp_params = [tf.Variable(param, trainable=False)
                  for param in var_list]
    # Define the swapping operation
    param_to_tmp = [tf.assign(tmp, param)
                    for tmp, param in safe_zip(tmp_params, var_list)]
    with tf.control_dependencies(param_to_tmp):
      avg_to_param = [tf.assign(param, avg)
                      for param, avg in safe_zip(var_list, avg_params)]
    with tf.control_dependencies(avg_to_param):
      tmp_to_avg = [tf.assign(avg, tmp)
                    for avg, tmp in safe_zip(avg_params, tmp_params)]
    swap = tmp_to_avg

  batch_size = args.batch_size

  assert batch_size % num_devices == 0
  device_batch_size = batch_size // num_devices

  if init_all:
    sess.run(tf.global_variables_initializer())
  else:
    initialize_uninitialized_global_variables(sess)

  # Check whether the hardware is working correctly

  # So far the failure has only been observed with 3 or more GPUs
  run_canary = run_canary and num_devices > 2
  if run_canary:
    canary_feed_dict = {}
    for x, y in safe_zip(preprocessed_xs, ys):
      canary_feed_dict[x] = x_train[:device_batch_size].copy()
      canary_feed_dict[y] = y_train[:device_batch_size].copy()
    # To reduce the runtime and memory cost of this canary,
    # we test the gradient of only one parameter.
    # For now this is just set to the first parameter in the list,
    # because it is an index that is always guaranteed to work.
    # If we think that this is causing false negatives and we should
    # test other parameters, we could test a random parameter from
    # the list or we could rewrite the canary to examine more than
    # one parameter.
    param_to_test = 0
    grad_vars = []
    for i in xrange(num_devices):
      dev_grads = grads[i]
      grad_vars.append(dev_grads[param_to_test][0])
    grad_values = sess.run(grad_vars, feed_dict=canary_feed_dict)
    failed = False
    for i in xrange(1, num_devices):
      if grad_values[0].shape != grad_values[i].shape:
        print("shape 0 does not match shape %d:" % i,
              grad_values[0].shape, grad_values[i].shape)
        failed = True
        continue
      if not np.allclose(grad_values[0], grad_values[i], atol=1e-6):
        print("grad_values[0]: ",
              grad_values[0].mean(), grad_values[0].max())
        print("grad_values[%d]: " %
              i, grad_values[i].mean(), grad_values[i].max())
        print("max diff: ", np.abs(
            grad_values[0] - grad_values[1]).max())
        failed = True
    if failed:
      print("Canary failed.")
      quit()

  for epoch in xrange(args.nb_epochs):
    # Indices to shuffle training set
    index_shuf = list(range(len(x_train)))
    # Randomly repeat a few training examples each epoch to avoid
    # having a too-small batch
    while len(index_shuf) % batch_size != 0:
      index_shuf.append(rng.randint(len(x_train)))
    nb_batches = len(index_shuf) // batch_size
    rng.shuffle(index_shuf)
    # Shuffling here versus inside the loop doesn't seem to affect
    # timing very much, but shuffling here makes the code slightly
    # easier to read
    x_train_shuffled = x_train[index_shuf]
    y_train_shuffled = y_train[index_shuf]

    prev = time.time()
    for batch in range(nb_batches):

      # Compute batch start and end indices
      start = batch * batch_size
      end = (batch + 1) * batch_size

      # Perform one training step
      feed_dict = {epoch_tf: epoch, batch_tf: batch}
      diff = end - start
      assert diff == batch_size
      for dev_idx in xrange(num_devices):
        cur_start = start + dev_idx * device_batch_size
        cur_end = start + (dev_idx + 1) * device_batch_size
        feed_dict[xs[dev_idx]] = x_train_shuffled[cur_start:cur_end]
        feed_dict[ys[dev_idx]] = y_train_shuffled[cur_start:cur_end]
      if cur_end != end:
        msg = ("batch_size (%d) must be a multiple of num_devices "
               "(%d).\nCUDA_VISIBLE_DEVICES: %s"
               "\ndevices: %s")
        args = (batch_size, num_devices,
                os.environ['CUDA_VISIBLE_DEVICES'],
                str(devices))
        raise ValueError(msg % args)
      if feed is not None:
        feed_dict.update(feed)

      _, loss_numpy = sess.run(
          [train_step, loss_value], feed_dict=feed_dict)

      if np.abs(loss_numpy) > loss_threshold:
        raise ValueError("Extreme loss during training: ", loss_numpy)
      if np.isnan(loss_numpy) or np.isinf(loss_numpy):
        raise ValueError("NaN/Inf loss during training")
    assert end == len(index_shuf)  # Check that all examples were used
    cur = time.time()
    _logger.info("Epoch " + str(epoch) + " took " +
                 str(cur - prev) + " seconds")
    if evaluate is not None:
      if use_ema:
        # Before running evaluation, load the running average
        # parameters into the live slot, so we can see how well
        # the EMA parameters are performing
        sess.run(swap)
      evaluate()
      if use_ema:
        # Swap the parameters back, so that we continue training
        # on the live parameters
        sess.run(swap)
  if use_ema:
    # When training is done, swap the running average parameters into
    # the live slot, so that we use them when we deploy the model
    sess.run(swap)

  return True


def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all
  towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been
     averaged across all towers.

  Modified from this tutorial: https://tinyurl.com/n3jr2vm
  """
  if len(tower_grads) == 1:
    return tower_grads[0]
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = [g for g, _ in grad_and_vars]

    # Average over the 'tower' dimension.
    grad = tf.add_n(grads) / len(grads)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    assert all(v is grad_and_var[1] for grad_and_var in grad_and_vars)
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
