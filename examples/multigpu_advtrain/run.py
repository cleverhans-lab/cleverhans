"""
This example adversarially trains a model using iterative attacks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import math
import time
import os
import logging
import random

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_loss
from cleverhans.utils_tf import batch_indices
from cleverhans.utils_tf import _FlagsWrapper
import cifar_input

from make_model import make_model
from evaluator import Evaluator, get_attack


FLAGS = flags.FLAGS
_data_path = {'svhn':  '/ssd1/datasets/svhn/',
              'cifar10':  '/ssd1/datasets/cifar-10/'}
if "CIFAR10_PARDIR" in os.environ:
    _data_path['cifar10'] = os.environ['CIFAR10_PARDIR']


def preprocess_batch(images_batch, preproc_func=None):
    if preproc_func is None:
        return images_batch

    with tf.variable_scope('preprocess'):
        images_list = tf.split(images_batch, int(images_batch.shape[0]))
        result_list = []
        for img in images_list:
            reshaped_img = tf.reshape(img, img.shape[1:])
            processed_img = preproc_func(reshaped_img)
            result_list.append(tf.expand_dims(processed_img, axis=0))
        result_images = tf.concat(result_list, axis=0)
    return result_images


def model_train(sess, model, x_pre, x, y, predictions, X_train, Y_train,
                predictions_adv=None, init_all=True, evaluate=None,
                verbose=True, args=None, hparams=None):
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
    :param init_all: (boolean) If set to true, all TF variables in the session
                     are (re)initialized, otherwise only previously
                     uninitialized variables are initialized before training.
    :param evaluate: function that is run after each training iteration
                     (typically to display the test/validation accuracy).
    :param verbose: (boolean) all print statements disabled when set to False.
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
                 If hparams.save is True, should also contain 'train_dir'
                 and 'filename'
    :return: True if model trained
    """
    args = _FlagsWrapper(args or {})

    # Check that necessary arguments were given (see doc above)
    assert args.nb_epochs, "Number of epochs was not given in args dict"
    assert args.learning_rate, "Learning rate was not given in args dict"
    assert args.batch_size, "Batch size was not given in args dict"

    if hparams.save:
        assert args.train_dir, "Directory for save was not given in args dict"
        assert args.filename, "Filename for save was not given in args dict"

    if hparams.model_type == 'resnet_tf':
        build_train_op = model.build_cost
    else:
        build_train_op = model_loss

    # Define loss
    with tf.variable_scope('train_loss'):
        loss = build_train_op(y, predictions)
        if predictions_adv is not None:
            if hparams.only_adv_train:
                loss = build_train_op(y, predictions_adv)
            else:
                adv_loss = build_train_op(y, predictions_adv)
                loss = (loss + adv_loss) / 2

    loss = tf.Print(loss, [loss])

    if hparams.model_type == 'resnet_tf':
        train_step = model.build_train_op_from_cost(loss)
    else:
        optim = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
        train_step = optim.minimize(loss)

    with sess.as_default():
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        evaluate.writer.add_graph(sess.graph)
        train_step_num = 0
        lrn_rate = hparams.learning_rate
        for epoch in six.moves.xrange(args.nb_epochs):
            if verbose:
                logging.info("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_train)

            # Indices to shuffle training set
            index_shuf = list(range(len(X_train)))
            random.shuffle(index_shuf)

            prev = time.time()
            for batch in range(nb_batches):
                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), args.batch_size)

                # update learning rate
                if hparams.model_type == 'resnet_tf':
                    if train_step_num < FLAGS.lrn_step:
                        lrn_rate = FLAGS.resnet_lrn
                    elif train_step_num < 30000:
                        lrn_rate = FLAGS.resnet_lrn/10
                    elif train_step_num < 35000:
                        lrn_rate = FLAGS.resnet_lrn/100
                    else:
                        lrn_rate = FLAGS.resnet_lrn/1000

                # Perform one training step
                fd = {x_pre: X_train[index_shuf[start:end]],
                      y: Y_train[index_shuf[start:end]]}
                if hparams.model_type == 'resnet_tf':
                    fd[model.lrn_rate] = lrn_rate

                train_step.run(feed_dict=fd)
                summary = tf.Summary()
                summary.value.add(tag='learning_rate',
                                  simple_value=lrn_rate)
                evaluate.writer.add_summary(summary, train_step_num)
                train_step_num += 1

                del fd

            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            if verbose:
                logging.info("\tEpoch took " + str(cur - prev) + " seconds")
            prev = cur
            if evaluate is not None:
                evaluate.evaluate()

            cond = ((epoch+1) % hparams.save_steps == 0
                    or epoch == args.nb_epochs)
            if hparams.save and cond:
                save_path = os.path.join(args.train_dir, args.filename)
                saver = tf.train.Saver()
                saver.save(sess, save_path)
                logging.info("Completed model training and saved at:" +
                             str(save_path))
            else:
                logging.info("Completed model training.")

    return True


def mnist_tutorial(hparams=None):
    """
    MNIST cleverhans tutorial
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """

    batch_size = hparams.batch_size

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    if FLAGS.dataset == 'mnist':
        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist()
        input_shape = (batch_size, 28, 28, 1)
        preproc_func = None
    elif FLAGS.dataset == 'cifar10':
        X_train, Y_train, X_test, Y_test = cifar_input.read_CIFAR10(
            _data_path[FLAGS.dataset])
        input_shape = (batch_size, 32, 32, 3)
        preproc_func = cifar_input.cifar_tf_preprocess

    # Define input TF placeholder
    x_pre = tf.placeholder(tf.float32, shape=input_shape, name='x')
    x = preprocess_batch(x_pre, preproc_func)
    y = tf.placeholder(tf.float32, shape=(batch_size, 10), name='y')

    # Use label smoothing
    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    with tf.device('/gpu:0'):
        model = make_model(input_shape=input_shape, **hparams.__flags)
        preds = model.fprop(x, training=True, bn_training=True)

    writer = tf.summary.FileWriter(hparams.save_dir, flush_secs=10)

    # Train an MNIST model
    train_params = {
        'nb_epochs': hparams.nb_epochs,
        'batch_size': hparams.batch_size,
        'learning_rate': hparams.learning_rate,
        'train_dir': hparams.save_dir,
        'filename': 'model.ckpt',
    }

    def init_eval():
        gid = 0
        with tf.device('/gpu:%d' % gid):
            evaluate = Evaluator(sess, model, batch_size, x_pre, x, y,
                                 (X_train, Y_train, X_test, Y_test),
                                 writer, hparams)
        return evaluate

    if not hparams.adv_train:
        logging.info("Naive training")

        evaluate = init_eval()
        model_train(sess, model, x_pre, x, y, preds, X_train, Y_train,
                    evaluate=evaluate,
                    args=train_params, hparams=hparams)
        evaluate(inc_epoch=False)
    else:
        logging.info("Adversarial training")

        evaluate = init_eval()
        logging.info("Initializing train attack %s" %
                     hparams.attack_type_train)
        adv2_x = get_attack(model, x, hparams.attack_type_train, sess,
                            batch_size=batch_size, y=y,
                            nb_iter=hparams.attack_nb_iter_train,
                            dataset=hparams.dataset)
        preds_2_adv = model.fprop(adv2_x, training=True, bn_training=False)

        # Perform and evaluate adversarial training
        model_train(sess, model, x_pre, x, y, preds, X_train, Y_train,
                    predictions_adv=preds_2_adv,
                    evaluate=evaluate,
                    args=train_params, hparams=hparams)

        evaluate(inc_epoch=False)

    writer.close()


def main(argv=None):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    hparams = flags.FLAGS

    mnist_tutorial(hparams=hparams)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_boolean('adv_train', False,
                         'Whether to do adversarial training')
    flags.DEFINE_boolean('save', True,
                         'Whether to save from a checkpoint')
    flags.DEFINE_string('save_dir', 'runs/X',
                        'Location to store logs/model.')
    flags.DEFINE_string('model_type', 'madry',
                        'Model type: madry|resnet_tf')
    flags.DEFINE_string('attack_type_train', 'PGDpgd_y',
                        'Attack type for adversarial training:\
                        FGSM|PGDpgd_y')
    flags.DEFINE_string('attack_type_test', 'FGSM',
                        'Attack type for test: FGSM|PGDpgd_y')
    flags.DEFINE_boolean('no_extra_tests', False,
                         'Disable some tests for debugging.')
    flags.DEFINE_string('dataset', 'mnist', 'Dataset mnist|cifar10|svhn')
    flags.DEFINE_boolean('only_adv_train', False,
                         'Do not train with clean examples when adv training.')
    flags.DEFINE_integer('save_steps', 50, 'Save model per X steps.')
    flags.DEFINE_integer('attack_nb_iter_train', None,
                         'Number of iterations of training attack')
    flags.DEFINE_integer('eval_iters', 1, 'Evaluate every X steps')
    flags.DEFINE_integer('lrn_step', 30000, 'Step to decrease learning rate')
    flags.DEFINE_float('resnet_lrn', .1, 'Initial learning rate for resnet')
    flags.DEFINE_string('optimizer', 'mom', 'Optimizer for resnet')
    flags.DEFINE_boolean('fast_tests', False, 'Fast tests against attacks')

    app.run()
