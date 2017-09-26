import tensorflow as tf

from cleverhans.utils import AccuracyReport
from cleverhans.utils_mnist import data_mnist
import svhn_input
import cifar_input
from utils import preprocess_batch

from make_model import make_model
from evaluator import Evaluator

_data_path = {'svhn':  '/ssd1/datasets/svhn/',
              'cifar10':  '/ssd1/datasets/cifar-10/'}


class Manager(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.init_session()
        self.init_data()
        self.init_inputs()
        self.init_model()
        self.init_eval()

    def init_session(self):
        # Set TF random seed to improve reproducibility
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
        x_pre, x, y = self.g0_inputs
        self.model.set_device('/gpu:0')
        self.evaluate = Evaluator(self.sess, self.model, self.batch_size,
                                  x_pre, x, y,
                                  self.data,
                                  self.writer,
                                  self.hparams)

    def eval(self):
        if self.evaluate is not None:
            self.evaluate.evaluate()

    def finish(self):
        self.writer.close()
        return self.report
