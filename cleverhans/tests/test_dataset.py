from cleverhans.dataset import Dataset


class LightweightDataset(Dataset):
    """
    A dataset that does not actually load any data so it is cheap to run
    in tests.
    """


def test_factory():
    d1 = LightweightDataset()
    factory = d1.get_factory()
    d2 = factory()
    assert type(d1) == type(d2)
