#!/usr/bin/env python3
"""
make_confidence_report_bundled.py
Usage:
  python make_confidence_report_bundled.py model.joblib

  where model.joblib is a file created by cleverhans.serial.save containing
  a picklable cleverhans.model.Model instance.

This script will run the model on clean data and bundled adversarial examples
( https://openreview.net/forum?id=H1g0piA9tQ ) for a max norm threat model
on the dataset the model was trained on.
It will save a ConfidenceReport to to model_bundled_report.joblib.
The report can be later loaded by another
script using cleverhans.serial.load.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from cleverhans.utils_tf import silence
# The silence() call must precede other imports in order to silence them.
# pylint does not like it but that's how it has to be.
# pylint: disable=C0413
silence()
from cleverhans.compat import flags
from cleverhans.confidence_report import make_confidence_report_bundled
from cleverhans.confidence_report import BATCH_SIZE
from cleverhans.confidence_report import TRAIN_START, TRAIN_END
from cleverhans.confidence_report import TEST_START, TEST_END
from cleverhans.confidence_report import WHICH_SET
from cleverhans.confidence_report import RECIPE
from cleverhans.confidence_report import REPORT_PATH


FLAGS = flags.FLAGS


def main(argv=None):
  """
  Make a confidence report and save it to disk.
  """
  try:
    _name_of_script, filepath = argv
  except ValueError:
    raise ValueError(argv)
  print(filepath)
  make_confidence_report_bundled(filepath=filepath,
                                 test_start=FLAGS.test_start,
                                 test_end=FLAGS.test_end,
                                 which_set=FLAGS.which_set,
                                 recipe=FLAGS.recipe,
                                 report_path=FLAGS.report_path, batch_size=FLAGS.batch_size)


if __name__ == '__main__':
  flags.DEFINE_integer('train_start', TRAIN_START, 'Starting point (inclusive)'
                       'of range of train examples to use')
  flags.DEFINE_integer('train_end', TRAIN_END, 'Ending point (non-inclusive) '
                       'of range of train examples to use')
  flags.DEFINE_integer('test_start', TEST_START, 'Starting point (inclusive) '
                       'of range of test examples to use')
  flags.DEFINE_integer('test_end', TEST_END, 'End point (non-inclusive) of '
                       'range of test examples to use')
  flags.DEFINE_string('recipe', RECIPE, 'Name of function from attack_bundling'
                      ' to run')
  flags.DEFINE_string('which_set', WHICH_SET, '"train" or "test"')
  flags.DEFINE_string('report_path', REPORT_PATH, 'Report path')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Batch size')
  tf.app.run()
