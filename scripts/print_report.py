#!/usr/bin/env python3
"""
print_report.py
Usage:
  print_report.py model_report.joblib
Prints out some basic statistics stored in a pickled ConfidenceReport
"""
import sys
import warnings

from cleverhans.confidence_report import ConfidenceReport
from cleverhans.serial import load


if len(sys.argv) == 2:
  # pylint doesn't realize that sys.argv will change at runtime
  # pylint:disable=unbalanced-tuple-unpacking
  _, path = sys.argv
else:
  raise ValueError("Wrong number of arguments")
the_report = load(path)

def current(report):
  """
  The current implementation of report printing.
  :param report: ConfidenceReport
  """
  if hasattr(report, "completed"):
    if report.completed:
      print("Report completed")
    else:
      print("REPORT NOT COMPLETED")
  else:
    warnings.warn("This report does not indicate whether it is completed. Support for reports without a `completed`"
                  "field may be dropped on or after 2019-05-11.")
  for key in report:
    covered = report[key].confidence > 0.5
    wrong = 1. - report[key].correctness
    failure_rate = (covered * wrong).mean()
    print(key, 'failure rate at t=.5', failure_rate)
    print(key, 'accuracy at t=0', report[key].correctness.mean())

def deprecated(report):
  """
  The deprecated implementation of report printing.
  :param report: dict
  """
  warnings.warn("Printing dict-based reports is deprecated. This function "
                "is included only to support a private development branch "
                "and may be removed without warning.")

  for key in report:
    confidence_name = 'confidence'
    correctness_name = 'correctness'
    if confidence_name not in report[key]:
      confidence_name = 'all_probs'
      correctness_name = 'correctness_mask'
      warnings.warn("'all_probs' is used only to temporarily support "
                    "the private development branch. This name can be "
                    "removed at any time without warning.")
    covered = report[key][confidence_name] > 0.5
    wrong = 1. - report[key][correctness_name]
    failure_rate = (covered * wrong).mean()
    print(key, 'failure rate at t=.5', failure_rate)
    print(key, 'accuracy at t=0', report[key][correctness_name].mean())


if isinstance(the_report, ConfidenceReport):
  current(the_report)
else:
  deprecated(the_report)
