#!/usr/bin/env python3
"""
make_confidence_report.py
Usage:
  python make_confidence_report_spsa.py model.joblib

  where model.joblib is a file created by cleverhans.serial.save containing
  a picklable cleverhans.model.Model instance.

This script will run the model on a variety of types of data and save a
ConfidenceReport to model_report.joblib.
The report can be later loaded by another script using cleverhans.serial.load.
This script puts the following entries in the report:
  clean : Clean data
  mc: MaxConfidence SPSA adversarial examples

This script works by running a single MaxConfidence attack on each example.
( https://openreview.net/forum?id=H1g0piA9tQ )
The MaxConfidence attack uses the SPSA optimizer.
This is not intended to be a generic strong attack; rather it is intended
to be a test for gradient masking.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
import tensorflow as tf

from cleverhans.utils_tf import silence
silence()
# The call to silence() must precede the other imports or they will log.
# pylint does not like this.
# pylint: disable=C0413
from cleverhans.attacks import SPSA
from cleverhans.attack_bundling import spsa_max_confidence_recipe
from cleverhans.serial import load
from cleverhans.utils import set_log_level
from cleverhans.compat import flags
from cleverhans.confidence_report import BATCH_SIZE
from cleverhans.confidence_report import TRAIN_START
from cleverhans.confidence_report import TRAIN_END
from cleverhans.confidence_report import TEST_START
from cleverhans.confidence_report import TEST_END
from cleverhans.confidence_report import WHICH_SET
from cleverhans.confidence_report import REPORT_PATH
NB_ITER_SPSA = 80
SPSA_SAMPLES = SPSA.DEFAULT_SPSA_SAMPLES


FLAGS = flags.FLAGS

def make_confidence_report_spsa(filepath, train_start=TRAIN_START,
                                train_end=TRAIN_END,
                                test_start=TEST_START, test_end=TEST_END,
                                batch_size=BATCH_SIZE, which_set=WHICH_SET,
                                report_path=REPORT_PATH,
                                nb_iter=NB_ITER_SPSA,
                                spsa_samples=SPSA_SAMPLES,
                                spsa_iters=SPSA.DEFAULT_SPSA_ITERS):
  """
  Load a saved model, gather its predictions, and save a confidence report.


  This function works by running a single MaxConfidence attack on each example,
  using SPSA as the underyling optimizer.
  This is not intended to be a strong generic attack.
  It is intended to be a test to uncover gradient masking.

  :param filepath: path to model to evaluate
  :param train_start: index of first training set example to use
  :param train_end: index of last training set example to use
  :param test_start: index of first test set example to use
  :param test_end: index of last test set example to use
  :param batch_size: size of evaluation batches
  :param which_set: 'train' or 'test'
  :param nb_iter: Number of iterations of PGD to run per class
  :param spsa_samples: Number of samples for SPSA
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.INFO)

  # Create TF session
  sess = tf.Session()

  if report_path is None:
    assert filepath.endswith('.joblib')
    report_path = filepath[:-len('.joblib')] + "_spsa_report.joblib"

  with sess.as_default():
    model = load(filepath)
  assert len(model.get_params()) > 0
  factory = model.dataset_factory
  factory.kwargs['train_start'] = train_start
  factory.kwargs['train_end'] = train_end
  factory.kwargs['test_start'] = test_start
  factory.kwargs['test_end'] = test_end
  dataset = factory()

  center = dataset.kwargs['center']
  center = np.float32(center)
  max_val = dataset.kwargs['max_val']
  max_val = np.float32(max_val)
  value_range = max_val * (1. + center)
  min_value = np.float32(0. - center * max_val)

  if 'CIFAR' in str(factory.cls):
    base_eps = 8. / 255.
  elif 'MNIST' in str(factory.cls):
    base_eps = .3
  else:
    raise NotImplementedError(str(factory.cls))

  eps = np.float32(base_eps * value_range)
  clip_min = min_value
  clip_max = max_val

  x_data, y_data = dataset.get_set(which_set)

  nb_classes = dataset.NB_CLASSES

  spsa_max_confidence_recipe(sess, model, x_data, y_data, nb_classes, eps,
                             clip_min, clip_max, nb_iter, report_path,
                             spsa_samples=spsa_samples,
                             spsa_iters=spsa_iters,
                             eval_batch_size=batch_size)

def main(argv=None):
  """
  Make a confidence report and save it to disk.
  """
  try:
    _name_of_script, filepath = argv
  except ValueError:
    raise ValueError(argv)
  make_confidence_report_spsa(filepath=filepath, test_start=FLAGS.test_start,
                              test_end=FLAGS.test_end,
                              which_set=FLAGS.which_set,
                              report_path=FLAGS.report_path,
                              nb_iter=FLAGS.nb_iter,
                              batch_size=FLAGS.batch_size,
                              spsa_samples=FLAGS.spsa_samples,
                              spsa_iters=FLAGS.spsa_iters)

if __name__ == '__main__':
  flags.DEFINE_integer('spsa_samples', SPSA_SAMPLES, 'Number samples for SPSA')
  flags.DEFINE_integer('spsa_iters', SPSA.DEFAULT_SPSA_ITERS,
                       'Passed to SPSA.generate')
  flags.DEFINE_integer('train_start', TRAIN_START, 'Starting point (inclusive)'
                       'of range of train examples to use')
  flags.DEFINE_integer('train_end', TRAIN_END, 'Ending point (non-inclusive) '
                       'of range of train examples to use')
  flags.DEFINE_integer('test_start', TEST_START,
                       'Starting point (inclusive) of range'
                       ' of test examples to use')
  flags.DEFINE_integer('test_end', TEST_END,
                       'End point (non-inclusive) of range'
                       ' of test examples to use')
  flags.DEFINE_integer('nb_iter', NB_ITER_SPSA, 'Number of iterations of SPSA')
  flags.DEFINE_string('which_set', WHICH_SET, '"train" or "test"')
  flags.DEFINE_string('report_path', REPORT_PATH, 'Path to save to')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Batch size for most jobs')
  tf.app.run()
