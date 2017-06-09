from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from cleverhans.attacks import Attack


class TestAttackClassInitArguments(unittest.TestCase):
    def test_model(self):
        import tensorflow as tf
        sess = tf.Session()

        # Exception is thrown when model does not have __call__ attribute
        with self.assertRaises(Exception) as context:
            model = tf.placeholder(tf.float32, shape=(None, 10))
            Attack(model, back='tf', sess=sess)
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

        # Test that it is permitted to provide no session
        Attack(model, back='tf', sess=None)

    def test_sess_generate_np(self):
        def model(x):
            return True

        class DummyAttack(Attack):
            def generate(self, x, **kwargs):
                return x

        attack = DummyAttack(model, back='tf', sess=None)
        with self.assertRaises(Exception) as context:
            attack.generate_np(0.)
        self.assertTrue(context.exception)

class TestParseParams(unittest.TestCase):
    def test_parse(self):
        def model():
            return True

        import tensorflow as tf
        sess = tf.Session()

        test_attack = Attack(model, back='tf', sess=sess)
        self.assertTrue(test_attack.parse_params({}))


if __name__ == '__main__':
    unittest.main()
