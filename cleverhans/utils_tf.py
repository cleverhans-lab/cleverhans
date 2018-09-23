from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import logging
import math
import numpy as np
import os
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.client import device_lib
import time
import warnings

from .utils import batch_indices, _ArgsWrapper, create_logger
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max, reduce_min
from cleverhans.compat import reduce_any
from cleverhans.compat import softmax_cross_entropy_with_logits

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
    warnings.warn('This function is deprecated.')
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
          init_all=True, evaluate=None, feed=None, args=None,
          rng=None, var_list=None, fprop_args=None, optimizer=None):
    """
    Train a TF graph.
    This function is not yet deprecated, but is likely to become deprecated
    soon. Prefer cleverhans.train.train when possible.
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
    args = _ArgsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    # Define accuracy symbolically
    if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
        correct_preds = tf.equal(tf.argmax(y, axis=-1),
                                 tf.argmax(predictions, axis=-1))
    else:
        correct_preds = tf.equal(tf.argmax(y, axis=tf.rank(y) - 1),
                                 tf.argmax(predictions,
                                           axis=tf.rank(predictions) - 1))

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


def tf_model_load(sess, file_path=None):
    """

    :param sess: the session object to restore
    :param file_path: path to the restored session, if None is
                      taken from FLAGS.train_dir and FLAGS.filename
    :return:
    """
    FLAGS = tf.app.flags.FLAGS
    with sess.as_default():
        saver = tf.train.Saver()
        if file_path is None:
            error = 'file_path argument is missing.'
            raise ValueError(error)
        saver.restore(sess, file_path)

    return True


def batch_eval(*args, **kwargs):
    # Inside function to avoid circul import
    from cleverhans.evaluation import batch_eval
    warnings.warn("batch_eval has moved to cleverhans.evaluation. "
                  "batch_eval will be removed from utils_tf on or after "
                  "2019-03-09.")


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
    with tf.name_scope(scope, "l2_batch_normalize") as scope:
        x_shape = tf.shape(x)
        x = tf.contrib.layers.flatten(x)
        x /= (epsilon + reduce_max(tf.abs(x), 1, keepdims=True))
        square_sum = reduce_sum(tf.square(x), 1, keepdims=True)
        x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
        x_norm = tf.multiply(x, x_inv_norm)
        return tf.reshape(x_norm, x_shape, scope)


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
    :param eps: Epilson, bound of the perturbation.
    """

    # Clipping perturbation eta to self.ord norm ball
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')
    reduc_ind = list(xrange(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if ord == 1:
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
        factor = tf.minimum(1., eps / norm)
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
    warnings.warn('This function is deprecated.')
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
            assert isinstance(device, str), type(device)
    return devices


def get_available_gpus():
    """
    Returns a list of string names of all available GPUs
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
