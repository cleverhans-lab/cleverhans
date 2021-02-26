"""Tests for cleverhans.dataset"""
from cleverhans.dataset import Dataset
from cleverhans.devtools.checks import CleverHansTest


class LightweightDataset(Dataset):
  """
  A dataset that does not actually load any data so it is cheap to run
  in tests.
  """


class TestDataset(CleverHansTest):
  """
  Tests for the Dataset class
  """

  def test_factory(self):
    """test_factory: Test that dataset->factory->dataset preserves type"""
    d1 = LightweightDataset()
    factory = d1.get_factory()
    d2 = factory()
    self.assertTrue(type(d1) is type(d2))
