from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from cleverhans.model import Model


class TestModelClass(unittest.TestCase):
    def test_default_graph_inference_state(self):
        # Define empty model
        model = Model()
        self.assertTrue(model.state == 'train')

    def test_change_graph_to_train(self):
        # Define empty model
        model = Model()

        # Set graph state to inference
        model.state = 'train'
        self.assertTrue(model.state == 'train')

    def test_get_layer(self):
        # Define empty model
        model = Model()
        x = []

        # Exception is thrown when `get_layer` not implemented
        with self.assertRaises(Exception) as context:
            model.get_layer(x, layer='')
        self.assertTrue(context.exception)

    def test_get_logits(self):
        # Define empty model
        model = Model()
        x = []

        # Exception is thrown when `get_logits` not implemented
        with self.assertRaises(Exception) as context:
            model.get_logits(x)
        self.assertTrue(context.exception)

    def test_get_probs(self):
        # Define empty model
        model = Model()
        x = []

        # Exception is thrown when `get_probs` not implemented
        with self.assertRaises(Exception) as context:
            model.get_probs(x)
        self.assertTrue(context.exception)

    def test_get_layer_names(self):
        # Define empty model
        model = Model()

        # Exception is thrown when `get_layer_names` not implemented
        with self.assertRaises(Exception) as context:
            model.get_layer_names()
        self.assertTrue(context.exception)

    def test_fprop(self):
        # Define empty model
        model = Model()
        x = []

        # Exception is thrown when `fprop` not implemented
        with self.assertRaises(Exception) as context:
            model.fprop(x)
        self.assertTrue(context.exception)


if __name__ == '__main__':
    unittest.main()
