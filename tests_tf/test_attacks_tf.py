"""Tests of cleverhans.attacks_tf

"""
import numpy as np
import tensorflow as tf
import unittest
from cleverhans.attacks_tf import fgm
from cleverhans.attacks_tf import compute_mask
from cleverhans.discretization_utils import discretize_uniform
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


class testComputeMask(unittest.TestCase):
    def test_compute_mask(self):
        eps = 1.0
        levels = 10
        np.random.seed(123)
        x = np.random.rand(10, 32, 32, 3)
        x_t = tf.constant(x, tf.float32)
        x_one_hot = discretize_uniform(x_t, levels,
                                       thermometer=False)
        x_thermometer = discretize_uniform(x_t, levels,
                                           thermometer=True)
        mask = [1] * 10
        mask = np.stack([mask] * 3, axis=1)
        mask = mask.flatten()
        mask = np.full((10, 32, 32, 30), mask)
        mask_o = compute_mask(
            levels,
            x - eps,
            x + eps,
            thermometer=False)
        mask_t = compute_mask(
            levels,
            x - eps,
            x + eps,
            thermometer=True)
        sess = tf.Session()
        self.assertTrue(np.all(mask == sess.run(mask_o)))
        self.assertTrue(np.all(mask == sess.run(mask_t)))


if __name__ == '__main__':
    unittest.main()
