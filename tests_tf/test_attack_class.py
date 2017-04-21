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

        # Exception is thrown when no session provided with TF
        with self.assertRaises(Exception) as context:
            Attack(model, back='tf', sess=None)
        self.assertTrue(context.exception)


if __name__ == '__main__':
    unittest.main()
