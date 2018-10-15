#!/usr/bin/env python
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
import sys

from matplotlib import pyplot

from cleverhans.plot.success_fail import plot_report_from_path

if __name__ == "__main__":
  report_paths = sys.argv[1:]

  for report_path in report_paths:
    plot_report_from_path(report_path, label=report_path)
  pyplot.legend()

  pyplot.xlim(-.01, 1.)
  pyplot.ylim(0., 1.)

  pyplot.show()
