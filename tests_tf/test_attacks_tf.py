"""Tests of cleverhans.attacks_tf

"""
import numpy as np
import tensorflow as tf
from cleverhans.attacks_tf import fgm
from cleverhans.devtools.mocks import random_feed_dict


def test_fgm_gradient_max():
    input_dim = 2
    num_classes = 3
    batch_size = 4
    rng = np.random.RandomState([2017, 8, 23])
    x = tf.placeholder(tf.float32, [batch_size, input_dim])
    weights = tf.placeholder(tf.float32, [input_dim, num_classes])
    logits = tf.matmul(x, weights)
    probs = tf.nn.softmax(logits)
    adv_x = fgm(x, probs)
    random_example = rng.randint(batch_size)
    random_feature = rng.randint(input_dim)
    output = tf.slice(adv_x, [random_example, random_feature], [1, 1])
    dx, = tf.gradients(output, x)
    # The following line catches GitHub issue #243
    assert dx is not None
    sess = tf.Session()
    dx = sess.run(dx, feed_dict=random_feed_dict(rng, [x, weights]))
    ground_truth = np.zeros((batch_size, input_dim))
    ground_truth[random_example, random_feature] = 1.
    assert np.allclose(dx, ground_truth), (dx, ground_truth)
