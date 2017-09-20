"""Utility functions for mocking up tests.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def random_feed_dict(rng, placeholders):
    """
    Returns random data to be used with `feed_dict`.
    :param rng: A numpy.random.RandomState instance
    :param placeholders: List of tensorflow placeholders
    :return: A dict mapping placeholders to random numpy values
    """

    output = {}

    for placeholder in placeholders:
        if placeholder.dtype != 'float32':
            raise NotImplementedError()
        value = rng.randn(*placeholder.shape).astype('float32')
        output[placeholder] = value

    return output
