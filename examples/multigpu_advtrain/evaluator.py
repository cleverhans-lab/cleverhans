import os
import numpy as np
import logging

import tensorflow as tf

from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MadryEtAl

from model import MLP_probs
from target_attacks import MadryEtAlMultiGPU


def create_adv_by_name(model, x, attack_type, sess, dataset, y=None, **kwargs):
    """
    Creates the symbolic graph of an adversarial example given the name of
    an attack. Default parameters are used unless a different value is given
    in kwargs.
    :param model: ?
    :param x: Symbolic input to the attack.
    :param attack_type: A string that is the name of an attack.
    :param sess: Tensorflow session.
    :param dataset: The name of the dataset as a string to use for default
                   params.
    :param y: (optional) a symbolic variable for the labels.
    :param kwargs: (optional) additional parameters to be passed to the attack.
    """
    attack_names = {'FGSM': FastGradientMethod,
                    'MadryEtAl': MadryEtAl,
                    'MadryEtAl_y': MadryEtAl,
                    'MadryEtAl_y_multigpu': MadryEtAlMultiGPU
                    }

    if attack_type not in attack_names:
        raise Exception('Attack %s not defined.' % attack_type)

    attack_params_shared = {
        'mnist': {'eps': .3, 'eps_iter': 0.01, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 40},
        'cifar10': {'eps': 8./255, 'eps_iter': 0.01, 'clip_min': 0.,
                    'clip_max': 1., 'nb_iter': 20}
    }

    with tf.variable_scope(attack_type):
        model = MLP_probs(model)
        attack_class = attack_names[attack_type]
        params = attack_params_shared[dataset]
        params.update(kwargs)
        attack = attack_class(model, sess=sess)
        adv_x = attack.generate(x, **params)

    return adv_x


class Evaluator(object):
    def __init__(self, sess, model, batch_size, x_pre, x, y,
                 data,
                 writer, hparams={}):
        self.preds = model.fprop(x, training=False)
        self.sess = sess
        self.batch_size = batch_size
        self.x_pre = x_pre
        self.x = x
        self.y = y
        self.X_train, self.Y_train, self.X_test, self.Y_test = data
        self.writer = writer
        self.hparams = hparams

        # Evaluate on a fixed subsampled set of the train data
        self.eval_params = {'batch_size': batch_size}

        # Load adversarial examples generated with the CleverHans tutorial
        # model
        bbox_path = 'bbox_%s.npy' % hparams.dataset
        if not hparams.no_extra_tests and os.path.isfile(bbox_path):
            self.bbox_np = np.load(bbox_path)
        else:
            self.bbox_np = None

        self.no_extra_tests = hparams.no_extra_tests

        self.epoch = 0

        self.attack_type_train = hparams.attack_type_train
        self.attack_type_test = []
        for att_type in hparams.attack_type_test.split(','):
            if att_type == '':
                continue
            if '_multi' in att_type:
                at = att_type.rsplit('_multi')[0]
                self.attack_type_test += ['%s_%d' % (at, i)
                                          for i in range(10, 100, 10)]
            else:
                self.attack_type_test += [att_type]
        self.attacks = {}

        # Initialize the attack object and graph
        for att_type in self.attack_type_test:
            logging.info('Intializing attack %s' % att_type)
            col_name = None
            adv_x = create_adv_by_name(model, x, att_type, sess,
                                       dataset=hparams.dataset, y=y)
            if isinstance(adv_x, tuple):
                adv_x, self.YY = adv_x

            preds_adv = model.fprop(adv_x, training=False)
            self.attacks[att_type] = (col_name, adv_x, preds_adv)
            # visualize adversarial image
            tf.summary.image(att_type, adv_x, max_outputs=10)
        self.sum_op = tf.summary.merge_all()

    def log_value(self, tag, val, desc='', print_stats=False,
                  print_only=False):
        if print_stats:
            valp = ''
            if not print_only:
                for i in range(0, 101, 10):
                    pr = np.percentile(val, i)
                    self.summary.value.add(tag=tag, simple_value=pr)
                    valp += ' %d: %.4f' % (i, pr)
            logging.info('%s (%s):\t %s' % (desc, tag, valp))
        else:
            logging.info('%s (%s): %.4f' % (desc, tag, val))
            if not print_only:
                self.summary.value.add(tag=tag, simple_value=val)

    def eval_train(self, sess, x, y, preds, X_train, Y_train):
        subsample_factor = 100
        X_train_subsampled = X_train[::subsample_factor]
        Y_train_subsampled = Y_train[::subsample_factor]
        acc_train = model_eval(sess, x, y, preds, X_train_subsampled,
                               Y_train_subsampled, args=self.eval_params)
        self.log_value('train_accuracy_subsampled', acc_train,
                       'Clean accuracy, subsampled train')

    def eval_test(self, sess, x, y, preds, X_test, Y_test):
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        acc = model_eval(sess, x, y, preds, X_test, Y_test,
                         args=self.eval_params)
        self.log_value('test_accuracy_natural', acc,
                       'Clean accuracy, natural test')

    def eval_advs(self, sess, x, y, preds_adv, X_test, Y_test, att_type,
                  col_name):
        # Evaluate the accuracy of the MNIST model on adversarial examples
        end = (len(X_test) // self.batch_size) * self.batch_size

        if self.hparams.fast_tests:
            end = 10*self.batch_size

        acc = model_eval(sess, x, y, preds_adv, X_test[:end], Y_test[:end],
                         args=self.eval_params)
        print_only = False
        cond = ('FGSM_' in att_type or 'BasicIter_' in att_type
                or 'PGD_' in att_type)
        if cond:
            print_only = True
        self.log_value('test_accuracy_%s' % att_type, acc,
                       'Test accuracy on adversarial examples',
                       print_only=print_only)
        return acc

    def eval_trans_advs(self, sess, x, y, preds, X_test, Y_test, bbox_np):
        # Evaluate on the transferred adversarial examples
        acc_bbox = model_eval(sess, x, y, preds, bbox_np,
                              Y_test[:len(bbox_np)], args=self.eval_params)
        self.log_value('test_accuracy_trans_advs', acc_bbox,
                       'Accuracy on transferred')

    def evaluate(self, inc_epoch=True):
        preds = self.preds
        sess = self.sess
        x = self.x_pre
        y = self.y
        X_train = self.X_train
        Y_train = self.Y_train
        X_test = self.X_test
        Y_test = self.Y_test
        writer = self.writer

        bbox_np = self.bbox_np

        self.summary = tf.Summary()
        if not self.no_extra_tests:
            self.eval_train(sess, x, y, preds, X_train, Y_train)
        self.eval_test(sess, x, y, preds, X_test, Y_test)

        acc = {}
        if self.epoch % self.hparams.eval_iters == 0:
            if not self.no_extra_tests:
                if self.bbox_np is not None:
                    self.eval_trans_advs(sess, x, y, preds, X_test, Y_test,
                                         bbox_np)

            for att_type in self.attack_type_test:
                col_name, adv_x, preds_adv = self.attacks[att_type]
                acc[att_type] = self.eval_advs(sess, x, y, preds_adv, X_test,
                                               Y_test, att_type, col_name)
        writer.add_summary(self.summary, self.epoch)
        if self.epoch % 20 == 0 and self.sum_op is not None:
            sm_val = sess.run(self.sum_op,
                              feed_dict={x: X_test[:self.batch_size],
                                         y: Y_test[:self.batch_size]})
            writer.add_summary(sm_val)

            for att_type in self.attack_type_test:
                if att_type.split('_')[-1].isdigit():
                    at, eps = att_type.rsplit('_')
                    acc_summary = tf.Summary()
                    acc_summary.value.add(tag=at, simple_value=acc[att_type])
                    writer.add_summary(acc_summary, eps)

        self.epoch += 1 if inc_epoch else 0

    def __call__(self, **kwargs):
        return self.evaluate(**kwargs)
