from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
import tensorflow as tf

from cleverhans import utils_tf


def numpy_kl_with_logits(q_logits, p_logits):
    def numpy_softmax(logits):
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    q = numpy_softmax(q_logits)
    p = numpy_softmax(p_logits)
    return (q * (np.log(q) - np.log(p))).sum(axis=1).mean()


class TestUtilsTF(unittest.TestCase):
    def test_l2_batch_normalize(self):
        with tf.Session() as sess:
            x = tf.random_normal((100, 1000))
            x_norm = sess.run(utils_tf.l2_batch_normalize(x))
            self.assertTrue(
                np.allclose(np.sum(x_norm**2, axis=1), 1, atol=1e-6))

    def test_kl_with_logits(self):
        q_logits = tf.placeholder(tf.float32, shape=(100, 20))
        p_logits = tf.placeholder(tf.float32, shape=(100, 20))
        q_logits_np = np.random.normal(0, 10, size=(100, 20))
        p_logits_np = np.random.normal(0, 10, size=(100, 20))
        with tf.Session() as sess:
            kl_div_tf = sess.run(utils_tf.kl_with_logits(q_logits, p_logits),
                                 feed_dict={q_logits: q_logits_np,
                                            p_logits: p_logits_np})
        kl_div_ref = numpy_kl_with_logits(q_logits_np, p_logits_np)
        self.assertTrue(np.allclose(kl_div_ref, kl_div_tf))


if __name__ == '__main__':
    unittest.main()
