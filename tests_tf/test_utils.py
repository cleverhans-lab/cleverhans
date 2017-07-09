from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from cleverhans import utils


class TestUtils(unittest.TestCase):
    def test_other_classes_neg_class_ind(self):
        with self.assertRaises(Exception) as context:
            utils.other_classes(10, -1)
        self.assertTrue(context.exception)

    def test_other_classes_invalid_class_ind(self):
        with self.assertRaises(Exception) as context:
            utils.other_classes(5, 8)
        self.assertTrue(context.exception)

    def test_other_classes_return_val(self):
        res = utils.other_classes(5, 2)
        res_expected = [0, 1, 3, 4]
        self.assertTrue(res == res_expected)


if __name__ == '__main__':
    unittest.main()
