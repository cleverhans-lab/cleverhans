"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

from cleverhans.attacks import FastGradientMethod
from cleverhans.defenses import LossXEntropy
from cleverhans.model import Model
from cleverhans.utils import AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def auto_pad(x, mod):
    xs = x.shape
    pad_h = xs[1] % (1 << mod)
    pad_w = xs[2] % (1 << mod)
    if pad_h == pad_w == 0:
        return
    return tf.pad(x, [[0] * 2, [pad_h // 2, (pad_h + 1) // 2],
                      [pad_w // 2, (pad_w + 1) // 2], [0] * 2])


class ModelBaseline(Model):
    def __init__(self, scope, nb_classes, depth, scales=3, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.depth = depth
        self.scales = scales

    def fprop(self, x, **kwargs):
        del kwargs
        act = tf.nn.relu
        my_conv = functools.partial(tf.layers.conv2d, kernel_size=3,
                                    padding='same', activation=act)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = auto_pad(x, 1 << self.scales)
            y = my_conv(y, self.depth)
            for scale in range(self.scales):
                y = my_conv(y, self.depth << scale)
                y = my_conv(y, self.depth << (scale + 1))
                y = tf.layers.average_pooling2d(y, 2, 2)
            y = my_conv(y, self.depth << self.scales)
            y = tf.layers.conv2d(y, self.nb_classes, 3, padding='same')
            logits = tf.reduce_mean(y, [1, 2])
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


def spectral_norm(kernel):
    """Computes the spectral norm of a weight matrix."""
    shape = kernel.shape.as_list()
    scope = kernel.name.split(':')[0]
    kernel = tf.reshape(kernel, [-1, shape[-1]])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        u = tf.get_variable('u', [1, shape[-1]],
                            initializer=tf.truncated_normal_initializer(),
                            trainable=False)
    v_ = tf.nn.l2_normalize(tf.matmul(u, kernel, transpose_b=True))
    u_ = tf.nn.l2_normalize(tf.matmul(v_, kernel))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, kernel), u_, transpose_b=True))
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(u, u_))
    return sigma


def my_conv2d_sn(inputs, *args, **kwargs):
    act = kwargs.get('activation', None)
    if 'activation' in kwargs:
        del kwargs['activation']
    layer = tf.layers.Conv2D(*args, **kwargs)
    y = layer.apply(inputs) / spectral_norm(layer.kernel)
    if act is not None:
        y = act(y)
    return y


class ModelBaselineSpectralNorm(ModelBaseline):
    def fprop(self, x, **kwargs):
        del kwargs
        act = tf.nn.relu
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = auto_pad(x, 1 << self.scales)
            y = my_conv2d_sn(y, self.depth, 3, padding='same', activation=act)
            for scale in range(self.scales):
                y = my_conv2d_sn(y, self.depth << scale, 3, padding='same',
                                 activation=act)
                y = my_conv2d_sn(y, self.depth << (scale + 1), 3,
                                 padding='same', activation=act)
                y = tf.nn.avg_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
            y = my_conv2d_sn(y, self.depth << self.scales, 3, padding='same',
                             activation=act)
            y = my_conv2d_sn(y, self.nb_classes, 3, padding='same')
            logits = tf.reduce_mean(y, [1, 2])
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


class ModelBasicCNN(Model):
    def __init__(self, scope, nb_classes, nb_filters, **kwargs):
        del kwargs
        Model.__init__(self, scope, nb_classes, locals())
        self.nb_filters = nb_filters

    def fprop(self, x, **kwargs):
        del kwargs
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            y = tf.layers.conv2d(x, self.nb_filters, 8, strides=2,
                                 padding='same', activation=tf.nn.relu)
            y = tf.layers.conv2d(y, self.nb_filters, 6, strides=2,
                                 padding='valid', activation=tf.nn.relu)
            y = tf.layers.conv2d(y, self.nb_filters, 5, strides=1,
                                 padding='valid', activation=tf.nn.relu)
            logits = tf.layers.dense(tf.layers.flatten(y), self.nb_classes)
            return {self.O_LOGITS: logits,
                    self.O_PROBS: tf.nn.softmax(logits=logits)}


def mnist_tutorial(nb_epochs=6, batch_size=128, learning_rate=0.0001,
                   clean_train=True, testing=False,
                   backprop_through_attack=False,
                   nb_filters=64):
    report = AccuracyReport()
    x_train, y_train, x_test, y_test = data_mnist(train_start=0,
                                                  train_end=60000,
                                                  test_start=0,
                                                  test_end=10000)

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    eval_params = {'batch_size': batch_size}
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    sess = tf.Session()

    if clean_train:
        model = ModelBasicCNN('model1', 10, nb_filters)
        preds = model.get_logits(x)
        loss = LossXEntropy(model, smoothing=0.1)

        def do_eval(preds, x_set, y_set, report_key, is_adv=None):
            acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
            setattr(report, report_key, acc)
            if is_adv is None:
                report_text = None
            elif is_adv:
                report_text = 'adversarial'
            else:
                report_text = 'legitimate'
            if report_text:
                print('Test accuracy on %s examples: %0.4f' %
                      (report_text, acc))

        def evaluate():
            do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

        model_train(sess, loss, x, y, x_train, y_train, evaluate=evaluate,
                    args=train_params, rng=rng, var_list=model.get_params())

        # Calculate training error
        if testing:
            do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_logits(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)

        # Calculate training error
        if testing:
            do_eval(preds_adv, x_train, y_train, 'train_clean_train_adv_eval')

        print('Repeating the process, using adversarial training')

    # Create a new model and train it to be robust to FastGradientMethod
    model2 = ModelBasicCNN('model2', 10, nb_filters)
    fgsm2 = FastGradientMethod(model2, sess=sess)

    def attack(x):
        return fgsm2.generate(x, **fgsm_params)

    loss2 = LossXEntropy(model2, smoothing=0.1, attack=attack)
    preds2 = model2.get_logits(x)
    adv_x2 = attack(x)

    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x2 = tf.stop_gradient(adv_x2)
    preds2_adv = model2.get_logits(adv_x2)

    def evaluate2():
        # Accuracy of adversarially trained model on legitimate test inputs
        do_eval(preds2, x_test, y_test, 'adv_train_clean_eval', False)
        # Accuracy of the adversarially trained model on adversarial examples
        do_eval(preds2_adv, x_test, y_test, 'adv_train_adv_eval', True)

    # Perform and evaluate adversarial training
    model_train(sess, loss2, x, y, x_train, y_train, evaluate=evaluate2,
                args=train_params, rng=rng, var_list=model2.get_params())

    # Calculate training errors
    if testing:
        do_eval(preds2, x_train, y_train, 'train_adv_train_clean_eval')
        do_eval(preds2_adv, x_train, y_train, 'train_adv_train_adv_eval')

    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
