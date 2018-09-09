from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import math
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf
import time
import warnings
import logging

from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max, reduce_min
from cleverhans.compat import reduce_any
from cleverhans.compat import softmax_cross_entropy_with_logits

from tensorflow.python.client import device_lib


_logger = create_logger("train")
_logger.setLevel(logging.INFO)


def train(sess, loss, x_train, y_train,
          init_all=True, evaluate=None, feed=None, args=None,
          rng=None, var_list=None, fprop_args=None, optimizer=None,
          devices=None):
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
    ys = []
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
            assert isinstance(device, str), type(device)
    for idx, device in enumerate(devices):
        with tf.device(device):
            x = tf.placeholder(x_train.dtype, (None,) + x_train.shape[1:])
            y = tf.placeholder(x_train.dtype, (None,) + y_train.shape[1:])
            xs.append(x)
            ys.append(y)

            loss_value = loss.fprop(x, y, **fprop_args)

            grads.append(optimizer.compute_gradients(
                loss_value, var_list=var_list))
    num_devices = len(devices)
    print("num_devices: ", num_devices)
    grad = avg_grads(grads)
    # Trigger update operations within the default graph (such as batch_norm).
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = optimizer.apply_gradients(grad)

    batch_size = args.batch_size

    with sess.as_default():
        if init_all:
            sess.run(tf.global_variables_initializer())
        else:
            initialize_uninitialized_global_variables(sess)

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
                # start, end = batch_indices(
                #    batch, len(x_train), args.batch_size)

                # Perform one training step
                feed_dict = {}
                diff = end - start
                assert diff == batch_size
                stride = diff // num_devices
                for dev_idx in xrange(num_devices):
                    cur_start = start + dev_idx * stride
                    cur_end = start + (dev_idx + 1) * stride
                    feed_dict[xs[dev_idx]
                              ] = x_train_shuffled[cur_start:cur_end]
                    feed_dict[ys[dev_idx]
                              ] = y_train_shuffled[cur_start:cur_end]
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
                sess.run(train_step, feed_dict=feed_dict)
            assert end == len(index_shuf)  # Check that all examples were used
            cur = time.time()
            _logger.info("Epoch " + str(epoch) + " took " +
                         str(cur - prev) + " seconds")
            if evaluate is not None:
                evaluate()

    return True


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
