from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
import tensorflow as tf

from cleverhans import attacks_tf

class TestAttacksTF(unittest.TestCase):
    def test_jsma_batch_with_feed(self):
        with tf.Session() as sess:
            X = np.random.rand(1, 13)

            # construct a simple graph that will require extra placeholders
            x = tf.placeholder('float', shape=(None, 13))
            keep_prob = tf.placeholder('float')
            W = tf.Variable(tf.random_normal([13, 10]))
            b = tf.Variable(tf.random_normal([10]))
            logits = tf.nn.dropout(tf.add(tf.matmul(x, W), b),
                                   keep_prob=keep_prob)
            sess.run(tf.global_variables_initializer())

            # jsma should work without generating an error
            jacobian = attacks_tf.jacobian_graph(logits, x, 10)
            attacks_tf.jsma_batch(sess, x, logits, jacobian, X, theta=1.,
                                  gamma=0.25, clip_min=0, clip_max=1,
                                  nb_classes=10, feed={keep_prob: 1.0})

if __name__ == '__main__':
    unittest.main()
