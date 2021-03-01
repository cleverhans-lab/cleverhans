"""Functionality for building tests.

We have to call this file "checks" and not anything with "test" as a
substring or nosetests will execute it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest

import numpy as np

class CleverHansTest(unittest.TestCase):
  """TestCase with some extra features"""

  def setUp(self):
    self.test_start = time.time()
    # seed the randomness
    np.random.seed(1234)

  def tearDown(self):
    print(self.id(), "took", time.time() - self.test_start, "seconds")

  def assertClose(self, x, y, *args, **kwargs):
    """Assert that `x` and `y` have close to the same value"""
    # self.assertTrue(np.allclose(x, y)) doesn't give a useful message
    # on failure
    assert np.allclose(x, y, *args, **kwargs), (x, y)
