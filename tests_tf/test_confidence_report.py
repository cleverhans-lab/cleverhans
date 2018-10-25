"""
Tests for cleverhans.confidence_report
"""

import numpy as np

from cleverhans.confidence_report import ConfidenceReport
from cleverhans.confidence_report import ConfidenceReportEntry


def test_confidence_report():
  # Test that we can make a confidence report, put an entry in it, and get
  # that entry back out
  report = ConfidenceReport()
  entry = ConfidenceReportEntry(correctness=np.array([True, False]),
                                confidence=np.array([0.9, 0.1]))
  report['clean'] = entry
  assert report['clean'] is entry
