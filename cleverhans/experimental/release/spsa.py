#!/usr/bin/env python
"""
spsa.py
Usage:
  python spsa.py model.joblib

  where model.joblib is a file created by cleverhans.serial.save containing
  a picklable cleverhans.model.Model instance.

This script will run the model on SPSA adversarial example for a max norm threat model
on the dataset the model was trained on.
It will save a report to to model_spsa_report.joblib.
The report can be later loaded by another
script using cleverhans.serial.load.

See cleverhans.confidence_report for more description of the report format.

Note that this script is targeting *misclassification* not *high confidence
misclassification*. Once an example is misclassified, the bundler will not
attempt to drive its confidence higher.
This means the evaluation runs faster, but it is only appropriate for accuracy
evaluations. It should not be used for success-fail curve evaluations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import time

import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_tf import silence
# The silence() call must precede other imports in order to silence them.
# pylint does not like it but that's how it has to be.
# pylint: disable=C0413
silence()
from cleverhans.attack_bundling import AttackConfig
from cleverhans.attack_bundling import BATCH_SIZE
from cleverhans.attack_bundling import bundle_attacks
from cleverhans.attack_bundling import Misclassify
from cleverhans.attack_bundling import num_devices
from cleverhans.attacks import SPSA
from cleverhans.confidence_report import make_confidence_report_bundled
from cleverhans.confidence_report import TRAIN_START, TRAIN_END
from cleverhans.confidence_report import TEST_START, TEST_END
from cleverhans.confidence_report import WHICH_SET
from cleverhans.confidence_report import REPORT_PATH
from cleverhans.confidence_report import BATCH_SIZE


FLAGS = flags.FLAGS

NB_ITER = 80
SPSA_ITERS = 1


def spsa_release_recipe(sess, model, x, y, nb_classes, eps, clip_min, clip_max, eps_iter, nb_iter, report_path,
                        eps_iter_small=None):
  """
  Runs SPSA with the goal of causing misclassification.

  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param nb_classes: int, number of classes
  :param eps: float, maximum size of perturbation (measured by max norm)
  :param eps_iter: ignored
  :param nb_iter: int, number of iterations for SPSA
  :param report_path: str, the path that the report will be saved to.
  :param eps_iter_small: ignored
  """
  spsa_attack = SPSA(model, sess)
  # Note: SPSA loss is a difference between the true class logit and the second biggest logit.
  # Its sign depends on whether the attack is targeted or not, but in all cases a threshold of 0
  # corresponds to attack success.
  # To make the attack strong, it is important to set the threshold to 0.
  # This is not just a speed optimization.
  # Without early stopping, the attack can "bounce off" the success region, so higher numbers
  # of iterations don't guarantee higher attack success rate. Early stopping with a threhsold of
  # 0. guarantees that if the attack ever enters the success region it will stop there.
  spsa_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max, "nb_iter" : nb_iter, "spsa_iters" : FLAGS.spsa_iters,
                 "spsa_samples": FLAGS.spsa_samples, "delta": FLAGS.delta, "learning_rate": FLAGS.learning_rate,
                 "early_stop_loss_threshold" : 0.}
  spsa_config = AttackConfig(spsa_attack, spsa_params, "spsa", True)
  attack_configs = [spsa_config]
  new_work_goal = {spsa_config: 1}
  goals = [Misclassify(new_work_goal=new_work_goal)]
  t1 = time.time()
  bundle_attacks(sess, model, x, y, attack_configs, goals, report_path, attack_batch_size=num_devices,
                 eval_batch_size=min(BATCH_SIZE, x.shape[0]))
  t2 = time.time()
  print("Took ", (t2 - t1) / 3600., "hours")


def main(argv=None):
  """
  Make a confidence report and save it to disk.
  """

  try:
    _name_of_script, filepath = argv
  except ValueError:
    raise ValueError(argv)
  assert filepath.endswith('.joblib')
  report_path = filepath[:-len('.joblib')] + "_spsa_report.joblib"
  make_confidence_report_bundled(filepath=filepath,
                                 test_start=FLAGS.test_start,
                                 test_end=FLAGS.test_end,
                                 which_set=FLAGS.which_set,
                                 recipe=spsa_release_recipe,
                                 report_path=report_path,
                                 nb_iter=FLAGS.nb_iter)


if __name__ == '__main__':
  flags.DEFINE_integer('train_start', TRAIN_START, 'Starting point (inclusive)'
                       'of range of train examples to use')
  flags.DEFINE_integer('train_end', TRAIN_END, 'Ending point (non-inclusive) '
                       'of range of train examples to use')
  flags.DEFINE_integer('test_start', TEST_START, 'Starting point (inclusive) of range'
                       ' of test examples to use')
  flags.DEFINE_integer('test_end', TEST_END, 'End point (non-inclusive) of range'
                       ' of test examples to use')
  flags.DEFINE_float('delta', SPSA.DEFAULT_DELTA, 'SPSA delta parameter')
  flags.DEFINE_float('learning_rate', SPSA.DEFAULT_LEARNING_RATE, 'SPSA learning rate parameter')
  flags.DEFINE_integer('spsa_samples', SPSA.DEFAULT_SPSA_SAMPLES, 'Number of samples for SPSA')
  flags.DEFINE_string('which_set', WHICH_SET, '"train" or "test"')
  flags.DEFINE_string('report_path', REPORT_PATH, 'Report path')
  flags.DEFINE_integer('nb_iter', NB_ITER, "Number of steps to run SPSA attack")
  flags.DEFINE_integer('spsa_iters', SPSA_ITERS, "Number of SPSA iterations")
  tf.app.run()
