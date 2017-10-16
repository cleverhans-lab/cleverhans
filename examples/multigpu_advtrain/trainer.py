import six
import math
import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from cleverhans.utils_tf import batch_indices
from cleverhans.utils import AccuracyReport
from cleverhans.utils_mnist import data_mnist
import cleverhans.utils_cifar as cifar_input
import cleverhans.utils_svhn as svhn_input
from utils import preprocess_batch

from make_model import make_model
from evaluator import Evaluator
from cleverhans.utils_tf import model_loss

import logging

_data_path = {'svhn':  '/ssd1/datasets/svhn/',
              'cifar10':  '/ssd1/datasets/cifar-10/'}


class TrainManager(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.evaluate = None
        self.feed_dict = {}
        self.step_num = 0
        self.init_session()
        self.init_data()
        self.init_inputs()
        self.init_model()
        self.create_train_graph()
        self.init_eval()

    def init_session(self):
        # Set TF random seed to improve reproducibility
        self.rng = np.random.RandomState([2017, 8, 30])
        tf.set_random_seed(1234)

        # Create TF session
        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True))

        # Object used to keep track of (and return) key accuracies
        self.report = AccuracyReport()
        self.writer = tf.summary.FileWriter(self.hparams.save_dir,
                                            flush_secs=10)

    def init_data(self):
        hparams = self.hparams
        batch_size = hparams.batch_size
        if hparams.dataset == 'mnist':
            # Get MNIST test data
            X_train, Y_train, X_test, Y_test = data_mnist()
            input_shape = (batch_size, 28, 28, 1)
            preproc_func = None
        elif hparams.dataset == 'cifar10':
            X_train, Y_train, X_test, Y_test = cifar_input.read_CIFAR10(
                _data_path[hparams.dataset])
            input_shape = (batch_size, 32, 32, 3)
            preproc_func = cifar_input.cifar_tf_preprocess
        elif hparams.dataset == 'svhn':
            X_train, Y_train, X_test, Y_test = svhn_input.read_SVHN(
                _data_path[hparams.dataset])
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

    def init_inputs(self):
        preproc_func = self.preproc_func
        input_shape = self.input_shape
        # Define input TF placeholder
        with tf.device('/gpu:0'):
            x_pre = tf.placeholder(tf.float32, shape=input_shape, name='x')
            x = preprocess_batch(x_pre, preproc_func)
            y = tf.placeholder(tf.float32, shape=(self.batch_size, 10),
                               name='y')
        self.g0_inputs = (x_pre, x, y)

    def init_model(self):
        hparams = self.hparams.__dict__['__flags']
        # Define TF model graph
        self.model = make_model(input_shape=self.input_shape, **hparams)
        self.model.set_device(None)
        return self.model

    def init_eval(self):
        logging.info("Init eval")
        x_pre, x, y = self.g0_inputs
        self.model.set_device('/gpu:0')
        self.evaluate = Evaluator(self.sess, self.model, self.batch_size,
                                  x_pre, x, y,
                                  self.data,
                                  self.writer,
                                  self.hparams)

    def eval(self):
        if self.evaluate is not None:
            self.evaluate.eval_multi()

    def finish(self):
        self.writer.close()
        return self.report

    def update_learning_params(self):
        model = self.model
        hparams = self.hparams
        fd = self.feed_dict
        step_num = self.step_num

        if hparams.model_type == 'resnet_tf':
            if step_num < hparams.lrn_step:
                lrn_rate = hparams.resnet_lrn
            elif step_num < 30000:
                lrn_rate = hparams.resnet_lrn/10
            elif step_num < 35000:
                lrn_rate = hparams.resnet_lrn/100
            else:
                lrn_rate = hparams.resnet_lrn/1000

            fd[model.lrn_rate] = lrn_rate

    def build_train_op(self, predictions, y, predictions_adv):
        model = self.model
        hparams = self.hparams
        if hparams.model_type == 'resnet_tf':
            build_train_op = model.build_cost
        else:
            build_train_op = model_loss

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
            optim = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
            train_step = optim.minimize(loss)

        return [train_step]

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
            self.init_tf(X_batch, Y_batch)

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
                    self.update_learning_params()

                    # train step
                    X_batch = X_train[index_shuf[start:end]]
                    Y_batch = Y_train[index_shuf[start:end]]

                    self.run(X_batch, Y_batch)
                    self.sync_params()

                # clean up the queue
                while not self.is_finished():
                    self.run()

                self.sync_params(forced=True)

                assert end >= len(X_train)  # Check that all examples were used
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
                    logging.info("Completed model training and saved at:" +
                                 str(save_path))
                else:
                    logging.info("Completed model training.")

    def init_tf(self, X_batch, Y_batch):
        x_pre, x, y = self.g0_inputs
        fd = {x_pre: X_batch, y: Y_batch}
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op, feed_dict=fd)

    def run_simple(self, X_batch=None, Y_batch=None):
        fetches, feed_dict = self.set_input(X_batch, Y_batch)
        fvals = self.sess.run(fetches, feed_dict=feed_dict)
        self.proc_fvals(fvals)
        self.step_num += 1

    def run_with_graph(self, X_batch, Y_batch):
        fetches, feed_dict = self.set_input(X_batch, Y_batch)

        self.writer.add_graph(self.sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        fvals = self.sess.run(fetches,
                              feed_dict=feed_dict,
                              options=run_options,
                              run_metadata=run_metadata)
        self.writer.add_run_metadata(run_metadata, 'graph')

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

        self.proc_fvals(fvals)
        self.step_num += 1

    def run(self, X_batch=None, Y_batch=None):
        if self.step_num == len(self.inputs)+1 and self.hparams.debug_graph:
            self.run_with_graph(X_batch, Y_batch)
        else:
            self.run_simple(X_batch, Y_batch)
