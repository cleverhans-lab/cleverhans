"""Tests for cleverhans.dataset"""
from cleverhans.dataset import Dataset


class LightweightDataset(Dataset):
  """
  A dataset that does not actually load any data so it is cheap to run
  in tests.
  """


def test_factory():
  """test_factory: Test that dataset->factory->dataset preserves type"""
  d1 = LightweightDataset()
  factory = d1.get_factory()
  d2 = factory()
  assert type(d1) is type(d2)
