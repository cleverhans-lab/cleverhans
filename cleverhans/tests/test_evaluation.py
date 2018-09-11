from cleverhans.evaluation import _CorrectFactory
from cleverhans.model import Model


def test_cache():
    # Test that _CorrectFactory can be cached
    model = Model()
    factory_1 = _CorrectFactory(model)
    factory_2 = _CorrectFactory(model)
    cache = {}
    cache[factory_1] = True
    assert factory_2 in cache
