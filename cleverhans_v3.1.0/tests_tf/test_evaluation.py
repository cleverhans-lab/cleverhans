"""Tests for cleverhans.evaluation"""
from cleverhans.devtools.checks import CleverHansTest
from cleverhans.evaluation import _CorrectFactory
from cleverhans.model import Model


class TestEvaluation(CleverHansTest):
  """Tests for cleverhans.evaluation"""

  def test_cache(self):
    """test_cache: Test that _CorrectFactory can be cached"""
    model = Model()
    factory_1 = _CorrectFactory(model)
    factory_2 = _CorrectFactory(model)
    cache = {}
    cache[factory_1] = True
    self.assertTrue(factory_2 in cache)
