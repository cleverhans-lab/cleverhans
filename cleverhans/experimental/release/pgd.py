#!/usr/bin/env python
"""
pgd.py
Usage:
  python pgd.py model.joblib

  where model.joblib is a file created by cleverhans.serial.save containing
  a picklable cleverhans.model.Model instance.

This script will run the model on clean data and PGD adversarial examples
for a max norm threat model
on the dataset the model was trained on.
It will save a report to to model_pgd_report.joblib.
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
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.confidence_report import make_confidence_report_bundled
from cleverhans.confidence_report import TRAIN_START, TRAIN_END
from cleverhans.confidence_report import TEST_START, TEST_END
from cleverhans.confidence_report import WHICH_SET
from cleverhans.confidence_report import RECIPE
from cleverhans.confidence_report import REPORT_PATH


FLAGS = flags.FLAGS

def pgd_release_recipe(sess, model, x, y, nb_classes, eps,
                       clip_min, clip_max, eps_iter, nb_iter,
                       report_path,
                       batch_size=BATCH_SIZE,
                       eps_iter_small=None):
  """
  Runs PGD, both targeted and untargeted, with two different step sizes,
  with the goal of causing misclassification.

  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param nb_classes: int, number of classes
  :param eps: float, maximum size of perturbation (measured by max norm)
  :param eps_iter: float, step size for one version of PGD attacks
    (will also run another version with eps_iter_small step size)
  :param nb_iter: int, number of iterations for the cheaper PGD attacks
    (will also run another version with 25X more iterations)
  :param report_path: str, the path that the report will be saved to.
  :param batch_size: int, the total number of examples to run simultaneously
  :param eps_iter_small: optional, float.
  """
  pgd_attack = ProjectedGradientDescent(model, sess)
  threat_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max}

  pgd_params = copy.copy(threat_params)
  pgd_params['nb_iter'] = nb_iter

  pgd_small_params = copy.copy(pgd_params)
  pgd_small_params['eps_iter'] = eps_iter_small

  pgd_large_params = copy.copy(pgd_params)
  pgd_large_params['eps_iter'] = eps_iter

  untargeted_small = AttackConfig(pgd_attack, pgd_small_params, "untargeted_small", True)
  untargeted_large = AttackConfig(pgd_attack, pgd_large_params, "untargeted_large", True)

  attack_configs = [untargeted_small, untargeted_large]
  
  assert batch_size % num_devices == 0
  dev_batch_size = batch_size // num_devices
  ones = tf.ones(dev_batch_size, tf.int32)
  for cls in range(nb_classes):
    cls_params_small = copy.copy(pgd_small_params)
    cls_params_large = copy.copy(pgd_large_params)
    cls_params_small['y_target'] = tf.to_float(tf.one_hot(ones * cls, nb_classes))
    cls_params_large['y_target'] = tf.to_float(tf.one_hot(ones * cls, nb_classes))
    cls_attack_small_config = AttackConfig(pgd_attack, cls_params_small, "target_" + str(cls) + "_small")
    cls_attack_large_config = AttackConfig(pgd_attack, cls_params_large, "target_" + str(cls) + "_large")
    attack_configs.extend([cls_attack_small_config, cls_attack_large_config])
  new_work_goal = {config: 1 for config in attack_configs}
  goals = [Misclassify(new_work_goal=new_work_goal)]
  bundle_attacks(sess, model, x, y, attack_configs, goals, report_path)


def main(argv=None):
  """
  Make a confidence report and save it to disk.
  """

  try:
    _name_of_script, filepath = argv
  except ValueError:
    raise ValueError(argv)
  assert filepath.endswith('.joblib')
  report_path = filepath[:-len('.joblib')] + "_pgd_" + str(FLAGS.nb_iter) + "_report.joblib"
  make_confidence_report_bundled(filepath=filepath,
                                 test_start=FLAGS.test_start,
                                 test_end=FLAGS.test_end,
                                 which_set=FLAGS.which_set,
                                 recipe=pgd_release_recipe,
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
  flags.DEFINE_string('recipe', RECIPE, 'Name of function from attack_bundling'
                      ' to run')
  flags.DEFINE_string('which_set', WHICH_SET, '"train" or "test"')
  flags.DEFINE_string('report_path', REPORT_PATH, 'Report path')
  flags.DEFINE_integer('nb_iter', 1000, 'Number of PGD iterations')
  tf.app.run()
