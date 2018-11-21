"""Runs CleverHans attacks on the Madry Lab CIFAR-10 challenge model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import logging

import tensorflow as tf
from tensorflow.python.platform import app, flags
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.utils_tf import model_eval
import cifar10_input


FLAGS = flags.FLAGS


def main(argv):

  model_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  if model_file is None:
    print('No model found')
    sys.exit()

  cifar = cifar10_input.CIFAR10Data(FLAGS.dataset_dir)

  nb_classes = 10
  X_test = cifar.eval_data.xs
  Y_test = to_categorical(cifar.eval_data.ys, nb_classes)
  assert Y_test.shape[1] == 10.

  set_log_level(logging.DEBUG)

  with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet
    model = make_wresnet()

    saver = tf.train.Saver()

    # Restore the checkpoint
    saver.restore(sess, model_file)

    nb_samples = FLAGS.nb_samples

    attack_params = {'batch_size': FLAGS.batch_size,
                     'clip_min': 0., 'clip_max': 255.}

    if FLAGS.attack_type == 'cwl2':
      from cleverhans.attacks import CarliniWagnerL2
      attacker = CarliniWagnerL2(model, sess=sess)
      attack_params.update({'binary_search_steps': 1,
                            'max_iterations': 100,
                            'learning_rate': 0.1,
                            'initial_const': 10,
                            'batch_size': 10
                            })

    else:  # eps and eps_iter in range 0-255
      attack_params.update({'eps': 8, 'ord': np.inf})
      if FLAGS.attack_type == 'fgsm':
        from cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, sess=sess)

      elif FLAGS.attack_type == 'pgd':
        attack_params.update({'eps_iter': 2, 'nb_iter': 20})
        from cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, sess=sess)

    eval_par = {'batch_size': FLAGS.batch_size}

    if FLAGS.sweep:
      max_eps = 16
      epsilons = np.linspace(1, max_eps, max_eps)
      for e in epsilons:
        t1 = time.time()
        attack_params.update({'eps': e})
        x_adv = attacker.generate(x, **attack_params)
        preds_adv = model.get_probs(x_adv)
        acc = model_eval(sess, x, y, preds_adv, X_test[
            :nb_samples], Y_test[:nb_samples], args=eval_par)
        print('Epsilon %.2f, accuracy on adversarial' % e,
              'examples %0.4f\n' % acc)
      t2 = time.time()
    else:
      t1 = time.time()
      x_adv = attacker.generate(x, **attack_params)
      preds_adv = model.get_probs(x_adv)
      acc = model_eval(sess, x, y, preds_adv, X_test[
          :nb_samples], Y_test[:nb_samples], args=eval_par)
      t2 = time.time()
      print('Test accuracy on adversarial examples %0.4f\n' % acc)
    print("Took", t2 - t1, "seconds")


if __name__ == '__main__':

  if "CIFAR10_CHALLENGE_DIR" in os.environ:
    cifar10_root = os.environ['CIFAR10_CHALLENGE_DIR']
  default_ckpt_dir = os.path.join(cifar10_root, 'models/adv_trained')
  default_data_dir = os.path.join(cifar10_root, 'cifar10_data')

  flags.DEFINE_integer('batch_size', 100, "Batch size")

  flags.DEFINE_integer('nb_samples', 1000, "Number of samples to test")

  flags.DEFINE_string('attack_type', 'fgsm', ("Attack type: 'fgsm'->'fast "
                                              "gradient sign method', "
                                              "'pgd'->'projected "
                                              "gradient descent', 'cwl2'->"
                                              "'Carlini & Wagner L2'"))
  flags.DEFINE_string('checkpoint_dir', default_ckpt_dir,
                      'Checkpoint directory to load')

  flags.DEFINE_string('dataset_dir', default_data_dir, 'Dataset directory')

  flags.DEFINE_bool('sweep', False, 'Sweep epsilon or single epsilon?')

  app.run(main)
