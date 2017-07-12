from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from cleverhans import utils


class TestUtils(unittest.TestCase):
    def test_to_categorical_no_nb_classes_arg(self):
        vec = np.asarray([0, 1, 2])
        cat = np.asarray([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        self.assertTrue(np.all(utils.to_categorical(vec) == cat))

    def test_to_categorical_with_nb_classes_arg(self):
        vec = np.asarray([0])
        cat = np.asarray([[1, 0, 0]])
        self.assertTrue(np.all(utils.to_categorical(vec, 3) == cat))

    def test_random_targets_vector(self):
        # Test utils.random_targets with a vector of labels as the input
        gt_labels = np.asarray([0, 1, 2, 3])
        rt = utils.random_targets(gt_labels, 5)

        # Make sure random_targets returns a one-hot encoded labels
        self.assertTrue(len(rt.shape) == 2)
        rt_labels = np.argmax(rt, axis=1)

        # Make sure all labels are different from the correct labels
        self.assertTrue(np.all(rt_labels != gt_labels))

    def test_random_targets_one_hot(self):
        # Test utils.random_targets with one-hot encoded labels as the input
        gt = np.asarray([[0, 0, 1, 0, 0],
                         [1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [1, 0, 0, 0, 0]])
        gt_labels = np.argmax(gt, axis=1)
        rt = utils.random_targets(gt, 5)

        # Make sure random_targets returns a one-hot encoded labels
        self.assertTrue(len(rt.shape) == 2)
        rt_labels = np.argmax(rt, axis=1)

        # Make sure all labels are different from the correct labels
        self.assertTrue(np.all(rt_labels != gt_labels))

    def test_random_targets_one_hot_single_label(self):
        # Test utils.random_targets with a single one-hot encoded label
        gt = np.asarray([0, 0, 1, 0, 0])
        gt = gt.reshape((1, 5))
        gt_labels = np.argmax(gt, axis=1)
        rt = utils.random_targets(gt, 5)

        # Make sure random_targets returns a one-hot encoded labels
        self.assertTrue(len(rt.shape) == 2)
        rt_labels = np.argmax(rt, axis=1)

        # Make sure all labels are different from the correct labels
        self.assertTrue(np.all(rt_labels != gt_labels))

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
