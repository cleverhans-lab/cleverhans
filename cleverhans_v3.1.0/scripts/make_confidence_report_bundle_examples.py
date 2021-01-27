#!/usr/bin/env python3
"""
make_confidence_report_bundle_examples.py
Usage:
  make_confidence_report_bundle_examples.py model.joblib a.npy
  make_confidence_report_bundle_examples.py model.joblib a.npy b.npy c.npy

  where model.joblib is a file created by cleverhans.serial.save containing
  a picklable cleverhans.model.Model instance and each examples_i.npy is
  a saved numpy array containing adversarial examples for a whole dataset.
  Usually example_i.npy is the output of make_confidence_report.py or
  make_confidence_report_bundled.py.

This script uses max-confidence attack bundling
( https://openreview.net/forum?id=H1g0piA9tQ )
to combine adversarial example datasets that were created earlier.
It will save a ConfidenceReport to to model_bundled_examples_report.joblib.
The report can be later loaded by another
script using cleverhans.serial.load.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import numpy as np
import tensorflow as tf

from cleverhans.utils_tf import silence
# We need to disable pylint's complaints about import order because `silence`
# works only if it is called before the other imports.
# pylint: disable=C0413
silence()
from cleverhans.attack_bundling import bundle_examples_with_goal, MaxConfidence
from cleverhans import serial
from cleverhans.compat import flags
from cleverhans.confidence_report import BATCH_SIZE
from cleverhans.confidence_report import TRAIN_START, TRAIN_END
from cleverhans.confidence_report import TEST_START, TEST_END
from cleverhans.confidence_report import WHICH_SET


FLAGS = flags.FLAGS


def main(argv=None):
  """
  Make a confidence report and save it to disk.
  """
  assert len(argv) >= 3
  _name_of_script = argv[0]
  model_filepath = argv[1]
  adv_x_filepaths = argv[2:]

  sess = tf.Session()
  with sess.as_default():
    model = serial.load(model_filepath)

  factory = model.dataset_factory
  factory.kwargs['train_start'] = FLAGS.train_start
  factory.kwargs['train_end'] = FLAGS.train_end
  factory.kwargs['test_start'] = FLAGS.test_start
  factory.kwargs['test_end'] = FLAGS.test_end
  dataset = factory()

  adv_x_list = [np.load(filepath) for filepath in adv_x_filepaths]
  x, y = dataset.get_set(FLAGS.which_set)
  for adv_x in adv_x_list:
    assert adv_x.shape == x.shape, (adv_x.shape, x.shape)
    # Make sure these were made for the right dataset with right scaling
    # arguments, etc.
    assert adv_x.min() >= 0. - dataset.kwargs['center'] * dataset.max_val
    assert adv_x.max() <= dataset.max_val
    data_range = dataset.max_val * (1. + dataset.kwargs['center'])

    if adv_x.max() - adv_x.min() <= .8 * data_range:
      warnings.warn("Something is weird. Your adversarial examples use "
                    "less than 80% of the data range."
                    "This might mean you generated them for a model with "
                    "inputs in [0, 1] and are now using them for a model "
                    "with inputs in [0, 255] or something like that. "
                    "Or it could be OK if you're evaluating on a very small "
                    "batch.")

  report_path = FLAGS.report_path
  if report_path is None:
    suffix = "_bundled_examples_report.joblib"
    assert model_filepath.endswith('.joblib')
    report_path = model_filepath[:-len('.joblib')] + suffix

  goal = MaxConfidence()
  bundle_examples_with_goal(sess, model, adv_x_list, y, goal,
                            report_path, batch_size=FLAGS.batch_size)


if __name__ == '__main__':
  flags.DEFINE_string('report_path', None, 'Report path')
  flags.DEFINE_integer('train_start', TRAIN_START, 'Starting point (inclusive)'
                       'of range of train examples to use')
  flags.DEFINE_integer('train_end', TRAIN_END, 'Ending point (non-inclusive) '
                       'of range of train examples to use')
  flags.DEFINE_integer('test_start', TEST_START, 'Starting point '
                       '(inclusive) of range of test examples to use')
  flags.DEFINE_integer('test_end', TEST_END, 'End point (non-inclusive) of '
                       'range of test examples to use')
  flags.DEFINE_string('which_set', WHICH_SET, '"train" or "test"')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'batch size')
  tf.app.run()
