#!/usr/bin/env python3
"""
Plots a success-fail curve ( https://openreview.net/forum?id=H1g0piA9tQ )
Usage:
plot_success_fail_curve.py model.joblib
plot_success_fail_curve.py model1.joblib model2.joblib

This script is mostly intended to rapidly visualize success-fail curves
during model development and testing.
To make nicely labeled plots formatted to fit the page / column of a
publication, you should probably write your own script that calls some
of the same plotting commands.
"""
from matplotlib import pyplot
import tensorflow as tf
from cleverhans.utils_tf import silence
silence()
# silence call must precede this imports. pylint doesn't like that
# pylint: disable=C0413
from cleverhans.compat import flags
from cleverhans.plot.success_fail import DEFAULT_FAIL_NAMES
from cleverhans.plot.success_fail import plot_report_from_path
FLAGS = flags.FLAGS

def main(argv=None):
  """Takes the path to a directory with reports and renders success fail plots."""
  report_paths = argv[1:]

  fail_names = FLAGS.fail_names.split(',')

  for report_path in report_paths:
    plot_report_from_path(report_path, label=report_path, fail_names=fail_names)
  pyplot.legend()

  pyplot.xlim(-.01, 1.)
  pyplot.ylim(0., 1.)

  pyplot.show()

if __name__ == '__main__':
  flags.DEFINE_string('fail_names', ','.join(DEFAULT_FAIL_NAMES), 'Names of adversarial datasets for failure rate')
  tf.app.run()
