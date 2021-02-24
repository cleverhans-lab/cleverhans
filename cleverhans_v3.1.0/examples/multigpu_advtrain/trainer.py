"""
This module provides Trainer classes that given a set of flags, create,
initialize and train a model. These classes use Runner objects to handle
multigpu/singlegpu training.
"""
# pylint: disable=missing-docstring
from collections import OrderedDict
import logging
import math
import time
import os

import numpy as np
import six
import tensorflow as tf

from cleverhans.utils_tf import batch_indices
from cleverhans.utils_mnist import data_mnist
import utils_cifar as cifar_input
import utils_svhn as svhn_input
from utils import preprocess_batch

from make_model import make_model
from evaluator import Evaluator
from evaluator import create_adv_by_name
from model import clone_variable


class TrainManager(object):
  """
  The base trainer class. Given an object of `hparams`, a trainer
  creates and initializes a model. After initialization, the method
  `model_train` can be used to train the model.
  """

  def __init__(self, hparams):
    """
    :param hparams: An instance of collections.namedtuple specifying the
                    model type and training configs. The parameters are
                    documented in `run_multigpu.py`.
    """
    self.hparams = hparams
    self.batch_size = hparams.batch_size
    self.evaluate = None
    self.step_num = 0
    self.report = None
    self._init_session()
    self._init_data()
    self._init_inputs()
    self._init_model()
    self._create_train_graph()
    self._init_eval()
    self.runner = None

  def _init_session(self):
    # Set TF random seed to improve reproducibility
    self.rng = np.random.RandomState([2017, 8, 30])
    tf.set_random_seed(1234)

    # Create TF session
    self.sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True))

    # Object used to keep track of (and return) key accuracies
    if self.hparams.save:
      self.writer = tf.summary.FileWriter(self.hparams.save_dir,
                                          flush_secs=10)
    else:
      self.writer = None

  def _init_data(self):
    hparams = self.hparams
    batch_size = hparams.batch_size
    if hparams.dataset == 'mnist':
      # Get MNIST test data
      X_train, Y_train, X_test, Y_test = data_mnist(
          train_start=hparams.train_start,
          train_end=hparams.train_end,
          test_start=hparams.test_start,
          test_end=hparams.test_end)
      input_shape = (batch_size, 28, 28, 1)
      preproc_func = None
    elif hparams.dataset == 'cifar10':
      X_train, Y_train, X_test, Y_test = cifar_input.read_CIFAR10(
          os.path.join(hparams.data_path, hparams.dataset))
      input_shape = (batch_size, 32, 32, 3)
      preproc_func = cifar_input.cifar_tf_preprocess
    elif hparams.dataset == 'svhn':
      X_train, Y_train, X_test, Y_test = svhn_input.read_SVHN(
          os.path.join(hparams.data_path, hparams.dataset))
      input_shape = (batch_size, 32, 32, 3)
      preproc_func = svhn_input.svhn_tf_preprocess

    # Use label smoothing
    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
    self.data = (X_train, Y_train, X_test, Y_test)
    self.input_shape = input_shape
    self.preproc_func = preproc_func

  def _init_inputs(self):
    preproc_func = self.preproc_func
    input_shape = self.input_shape
    # Define input TF placeholder
    with tf.device('/gpu:0'):
      x_pre = tf.placeholder(tf.float32, shape=input_shape, name='x')
      x = preprocess_batch(x_pre, preproc_func)
      y = tf.placeholder(tf.float32, shape=(self.batch_size, 10),
                         name='y')

    self.g0_inputs = {'x_pre': x_pre, 'x': x, 'y': y}

  def _init_model(self):
    flags = self.hparams.__dict__
    # Define TF model graph
    model = make_model(input_shape=self.input_shape, **flags)
    model.set_device(None)
    self.model = model

  def _init_eval(self):
    logging.info("Init eval")
    x_pre, x, y = [self.g0_inputs[k] for k in ['x_pre', 'x', 'y']]
    self.model.set_device('/gpu:0')
    self.evaluate = Evaluator(self.sess, self.model, self.batch_size,
                              x_pre, x, y,
                              self.data,
                              self.writer,
                              self.hparams)

  def eval(self, **kwargs):
    if self.evaluate is not None:
      self.report = self.evaluate.eval_multi()

  def finish(self):
    if self.writer:
      self.writer.close()
    return self.report

  def _update_learning_params(self):
    model = self.model
    hparams = self.hparams
    fd = self.runner.feed_dict
    step_num = self.step_num

    if hparams.model_type == 'resnet_tf':
      if step_num < hparams.lrn_step:
        lrn_rate = hparams.mom_lrn
      elif step_num < 30000:
        lrn_rate = hparams.mom_lrn/10
      elif step_num < 35000:
        lrn_rate = hparams.mom_lrn/100
      else:
        lrn_rate = hparams.mom_lrn/1000

      fd[model.lrn_rate] = lrn_rate

  def _build_train_op(self, predictions, y, predictions_adv):
    model = self.model
    hparams = self.hparams
    if hparams.model_type == 'resnet_tf':
      build_train_op = model.build_cost
    else:
      raise NotImplementedError("this configuration of this example is no longer maintained")

    # Define loss
    with tf.variable_scope('train_loss'):
      if predictions_adv is not None:
        if hparams.only_adv_train:
          loss = build_train_op(y, predictions_adv)
        else:
          loss = build_train_op(y, predictions)
          adv_loss = build_train_op(y, predictions_adv)
          loss = (loss + adv_loss) / 2
      else:
        loss = build_train_op(y, predictions)

    if hparams.model_type == 'resnet_tf':
      train_step = model.build_train_op_from_cost(loss)
    else:
      optim = tf.train.AdamOptimizer(learning_rate=hparams.adam_lrn)
      train_step = optim.minimize(loss)

    return train_step

  def model_train(self):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param hparams.save: boolean controlling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param evaluate: function that is run after each training iteration
                    (typically to display the test/validation accuracy).
    """

    assert self.runner is not None, (
        """Runner is not initialized. TrainerSingleGPU or TrainerMultiGPU
            instantiate a Runner object at initialization time.""")
    hparams = self.hparams
    batch_size = hparams.batch_size
    nb_epochs = hparams.nb_epochs
    train_dir = hparams.save_dir
    filename = 'model.ckpt'
    X_train = self.X_train
    Y_train = self.Y_train

    sess = self.sess

    with sess.as_default():
      X_batch = X_train[:batch_size]
      Y_batch = Y_train[:batch_size]
      self._init_tf(X_batch, Y_batch)

      for epoch in six.moves.xrange(nb_epochs):
        logging.info("Epoch " + str(epoch))

        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
        assert nb_batches * batch_size >= len(X_train)

        # Indices to shuffle training set
        index_shuf = list(range(len(X_train)))
        self.rng.shuffle(index_shuf)

        prev = time.time()
        for batch in range(nb_batches):
          # Compute batch start and end indices
          start, end = batch_indices(
              batch, len(X_train), batch_size)

          # Perform one training step
          self._update_learning_params()

          # Train step
          X_batch = X_train[index_shuf[start:end]]
          Y_batch = Y_train[index_shuf[start:end]]

          self._run({'x_pre': X_batch, 'y': Y_batch})
          self._sync_params()

        # Clean up the queue
        while not self.runner.is_finished():
          self._run()

        self._sync_params(forced=True)

        assert end >= len(X_train), (
            'Not all training examples are used.')
        cur = time.time()
        logging.info("\tEpoch took " + str(cur - prev) + " seconds")
        prev = cur

        self.eval()

        # Save model
        cond = ((epoch+1) % hparams.save_steps == 0
                or epoch == nb_epochs)
        if hparams.save and cond:
          save_path = os.path.join(train_dir, filename)
          saver = tf.train.Saver()
          saver.save(sess, save_path)
          logging.info("Model saved at: " + str(save_path))
    logging.info("Completed model training.")

  def _init_tf(self, X_batch, Y_batch):
    x_pre = self.g0_inputs['x_pre']
    y = self.g0_inputs['y']
    fd = {x_pre: X_batch, y: Y_batch}
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op, feed_dict=fd)

  def _run(self, X_batch=None):
    last_fvals = self.runner.run(X_batch)
    self.step_num += 1
    return last_fvals

  def _sync_params(self, forced=False):
    raise NotImplementedError('sync_params should be implemented.')

  def _create_train_graph(self):
    """
    The evaluation graph must be initialized after the train graph is
    fully initialized, otherwise, some of the variables will be created
    untrainable.
    """
    assert self.evaluate is None, ("""Evaluation graph should be initialzed
                                       after the train graph""")


class TrainerMultiGPU(TrainManager):
  """
  This class uses a `RunnerMultiGPU` object to train a model on multiple
  GPUs. It mainly overrides the `_create_train_graph` to create a graph
  for adversarial training on multiple GPUs.
  """

  def __init__(self, *args, **kwargs):
    super(TrainerMultiGPU, self).__init__(*args, **kwargs)
    from runner import RunnerMultiGPU
    self.runner = RunnerMultiGPU(self.inputs, self.outputs, sess=self.sess)

  def clone_g0_inputs_on_ngpus(self, inputs, outputs, g0_inputs):
    """
    Clone variables unused by the attack on all GPUs. Specifically, the
    ground-truth label, y, has to be preserved until the training step.

    :param inputs: A list of dictionaries as the inputs to each step.
    :param outputs: A list of dictionaries as the outputs of each step.
    :param g0_inputs: Initial variables to be cloned.
    :return: Updated inputs and outputs.
    """
    assert len(inputs) == len(outputs), (
        'Inputs and outputs should have the same number of elements.')

    inputs[0].update(g0_inputs)
    outputs[0].update(g0_inputs)

    # Copy g0_inputs forward
    for i in range(1, len(inputs)):
      # Create the graph for i'th step of attack
      device_name = inputs[i]['x'].device
      with tf.device(device_name):
        with tf.variable_scope('step%d' % i):
          for k, v in g0_inputs.iteritems():
            if k not in inputs[i]:
              v_copy = clone_variable(k, v)
              inputs[i][k] = v_copy
              outputs[i][k] = v_copy

    return inputs, outputs

  def _create_train_graph(self):
    super(TrainerMultiGPU, self)._create_train_graph()
    assert '_multigpu' in self.hparams.attack_type_train

    hparams = self.hparams
    model = self.model
    sess = self.sess

    # Create trainable variables on last gpu.
    # Variables are set to trainable or non-trainable first time they are
    # created. This caused a bug when the last gpu is used both for attack
    # generation and training. With this bug the result of naive training
    # was affected by the length of the unused adversarial generation
    # graph.
    device_name = '/gpu:%d' % (hparams.ngpu-1)
    model.set_device(device_name)
    with tf.device(device_name):
      x = clone_variable('x', self.g0_inputs['x'])
      model.set_training(training=True)
      preds = model.get_probs(x)

    # Generates steps on gpus
    model.set_training(training=False)
    logging.info("Initializing train attack %s" %
                 hparams.attack_type_train)
    inputs, outputs = create_adv_by_name(
        model, self.g0_inputs['x'], hparams.attack_type_train,
        sess, y=self.g0_inputs['y'], nb_iter=hparams.attack_nb_iter_train,
        dataset=hparams.dataset, ngpu=hparams.ngpu)

    inputs, outputs = self.clone_g0_inputs_on_ngpus(
        inputs, outputs, self.g0_inputs)

    # Train step on last gpu
    device_name = '/gpu:%d' % (hparams.ngpu-1)
    model.set_device(device_name)
    with tf.device(device_name):
      with tf.variable_scope('last'):
        inputs += [OrderedDict()]
        for k, v in outputs[-1].iteritems():
          v_copy = clone_variable(k, v)
          inputs[-1][k] = v_copy
        x = inputs[-1]['x']
        adv_x = inputs[-1]['adv_x']
        y = inputs[-1]['y']
        if not hparams.adv_train:
          model.set_training(training=True)
          preds = model.get_probs(x)
          preds_adv = None
        elif not hparams.only_adv_train:
          model.set_training(training=True)
          preds = model.get_probs(x)
          model.set_training(training=True)
          preds_adv = model.get_probs(adv_x)
        else:
          preds = None
          model.set_training(training=True)
          preds_adv = model.get_probs(adv_x)
        train_fetches = self._build_train_op(preds, y, preds_adv)

    outputs += [{'fetches': train_fetches}]

    # Create the sync operation
    device_name = '/gpu:%d' % (hparams.ngpu-1)
    model.set_device(device_name)
    with tf.device(device_name):
      sync_ops = model.create_sync_ops(host_device=device_name)

    self.inputs = inputs
    self.outputs = outputs
    self.sync_ops = sync_ops

  def _sync_params(self, forced=False):
    if forced or (self.step_num % self.hparams.sync_step == 0):
      self.sess.run(self.sync_ops)


class TrainerSingleGPU(TrainManager):
  """
  This class uses a `RunnerSingleGPU` object to train a model on a single
  GPU.
  """

  def __init__(self, *args, **kwargs):
    super(TrainerSingleGPU, self).__init__(*args, **kwargs)
    from runner import RunnerSingleGPU
    self.runner = RunnerSingleGPU(self.inputs, self.outputs,
                                  sess=self.sess)

  def _create_train_graph(self):
    super(TrainerSingleGPU, self)._create_train_graph()
    self.model.set_device('/gpu:0')
    hparams = self.hparams
    model = self.model
    x = self.g0_inputs['x']
    y = self.g0_inputs['y']
    sess = self.sess

    # Create trainable variables.
    model.set_training(training=True)
    preds = model.get_probs(x)

    if not hparams.adv_train:
      logging.info("Naive training")

      model.set_training(training=True)
      preds = model.get_probs(x)
      preds_adv = None
    else:
      logging.info("Adversarial training")
      logging.info("Initializing train attack %s" %
                   hparams.attack_type_train)

      model.set_training(training=False)
      adv_x = create_adv_by_name(
          model, x, hparams.attack_type_train, sess,
          y=y, nb_iter=hparams.attack_nb_iter_train,
          dataset=hparams.dataset)
      if hparams.only_adv_train:
        preds = None
        model.set_training(training=True)
        preds_adv = model.get_probs(adv_x)
      else:
        model.set_training(training=True)
        preds = model.get_probs(x)
        model.set_training(training=True)
        preds_adv = model.get_probs(adv_x)
    train_fetches = self._build_train_op(preds, y, preds_adv)

    self.inputs = [self.g0_inputs]
    self.outputs = [train_fetches]

  def _sync_params(self, forced=False):
    """
    Nothing to sync on single GPU.
    """
    return True
