from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from cleverhans.attacks import Attack


class TestAttackClassInitArguments(unittest.TestCase):
    def test_model(self):
        import theano.tensor as T

        # Exception is thrown when model does not have __call__ attribute
        with self.assertRaises(Exception) as context:
            model = T.matrix('y')
            Attack(model, back='th', sess=None)
        self.assertTrue(context.exception)

    def test_back(self):
        # Define empty model
        def model():
            return True

        # Exception is thrown when back is not tf or th
        with self.assertRaises(Exception) as context:
            Attack(model, back='test', sess=None)
        self.assertTrue(context.exception)

    def test_sess(self):
        # Define empty model
        def model():
            return True

        # Exception is thrown when session provided with TH
        with self.assertRaises(Exception) as context:
            Attack(model, back='th', sess=1)
        self.assertTrue(context.exception)


class TestAttackGenerate(unittest.TestCase):
    def test_inf_loop(self):
        def model(x):
            return x

        import theano.tensor as T
        x = T.tensor4('x')
        test_attack = Attack(model, back='th', sess=None)

        with self.assertRaises(Exception) as context:
            test_attack.generate(x)
        self.assertTrue(context.exception)


class TestAttackGenerateNp(unittest.TestCase):
    def test_inf_loop(self):
        def model(x):
            return x

        import numpy as np
        x_val = np.zeros((10, 5, 5, 1))

        test_attack = Attack(model, back='th', sess=None)
        with self.assertRaises(Exception) as context:
            test_attack.generate_np(x_val)
        self.assertTrue(context.exception)


class TestParseParams(unittest.TestCase):
    def test_parse(self):
        def model():
            return True

        test_attack = Attack(model, back='th', sess=None)
        self.assertTrue(test_attack.parse_params({}))


if __name__ == '__main__':
    unittest.main()
