from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
import tensorflow as tf

from cleverhans import discretization_utils
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper


class TestDiscretizationUtils(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        self.sess = tf.Session()

    def testFlattenLast(self):
        a = np.random.randint(0, 5, (10, 32, 32, 3, 20))
        b = np.reshape(a, (10, 32, 32, 3 * 20))
        a_t = tf.constant(a, dtype=tf.float32)
        flattened_a = discretization_utils.flatten_last(a_t)
        self.assertTrue(np.all(b == self.sess.run(flattened_a)))

    def testUnflattenLast(self):
        a = np.random.randint(0, 5, (10, 32, 32, 3 * 20))
        b = np.reshape(a, (10, 32, 32, 3, 20))
        a_t = tf.placeholder(dtype=tf.float32,
                             shape=[10, 32, 32, 3 * 20])
        unflattened_a = discretization_utils.unflatten_last(a_t, 20)
        self.assertTrue(np.all(b == self.sess.run(unflattened_a,
                                                  feed_dict={a_t: a})))

    def testOneHotToThermometer(self):
        one_hot_inputs = [[1] + [0] * 9, [0, 0, 1] + [0] * 7,
                          [0, 1] + [0] * 8, [0] * 9 + [1]]
        thermometer_inputs = [[1] + [0] * 9, [1, 1, 1] + [0] * 7,
                              [1, 1] + [0] * 8, [1] * 10]
        for i in range(len(one_hot_inputs)):
            one_hot_input = one_hot_inputs[i]
            thermometer_input = thermometer_inputs[i]
            levels = len(one_hot_input)
            one_hot = np.full((10, 32, 32, 3, levels), one_hot_input)
            x = tf.constant(one_hot, dtype=tf.int32)
            thermometer = np.full((10, 32, 32, 3, levels), thermometer_input)
            thermometer_x = discretization_utils.one_hot_to_thermometer(
                x, levels, flattened=False)
            self.assertTrue(np.all(thermometer ==
                                   self.sess.run(thermometer_x)))

    def testThermometerToOneHot(self):
        one_hot_inputs = [[1] + [0] * 9, [0, 1] + [0] * 8,
                          [0, 0, 1] + [0] * 7, [0] * 9 + [1]]
        thermometer_inputs = [[1] + [0] * 9, [1, 1] + [0] * 8,
                              [1, 1, 1] + [0] * 7, [1] * 10]
        for i in range(len(one_hot_inputs)):
            one_hot_input = one_hot_inputs[i]
            thermometer_input = thermometer_inputs[i]
            levels = len(one_hot_input)
            one_hot = np.full((10, 32, 32, 3, levels), one_hot_input)
            thermometer = np.full((10, 32, 32, 3, levels), thermometer_input)
            x = tf.constant(thermometer, dtype=tf.int32)
            one_hot_x = discretization_utils.thermometer_to_one_hot(
                x, levels, flattened=False)
            self.assertTrue(np.all(one_hot == self.sess.run(one_hot_x)))

    def testQuantizeUniformOneHot(self):
        values = [.1, .15, .95]
        buckets = [[1] + 9 * [0], [0, 1] + [0] * 8, 9 * [0] + [1]]
        for i in range(len(values)):
            bucket = buckets[i]
            value = values[i]
            levels = len(bucket)
            image = np.full((10, 32, 32, 3), value)
            image_t = tf.constant(image, dtype=tf.float32)
            one_hot = np.full((10, 32, 32, 3, levels), bucket)
            one_hot = np.reshape(one_hot, (10, 32, 32, 3 * levels))
            discretized_x = discretization_utils.discretize_uniform(image_t,
                                                                    levels)
            self.assertTrue(np.all(one_hot == self.sess.run(discretized_x)))

    def testQuantizeUniformThermometer(self):
        values = [.1, .15, .95]
        buckets = [[1] + [0] * 9, [1, 1] + [0] * 8, [1] * 10]
        for i in range(len(values)):
            bucket = buckets[i]
            value = values[i]
            levels = len(bucket)
            image = np.full((10, 32, 32, 3), value)
            image_t = tf.placeholder(dtype=tf.float32, shape=[10, 32, 32, 3])
            thermometer = np.full((10, 32, 32, 3, levels), bucket)
            thermometer = np.reshape(thermometer, (10, 32, 32, 3 * levels))
            discretized_x = discretization_utils.discretize_uniform(
                image_t, levels, thermometer=True)
            self.assertTrue(np.all(thermometer == self.sess.run(
                discretized_x,
                feed_dict={image_t: image})))

    def testQuantizeOneHotCounts(self):
        image = np.random.uniform(low=0., high=1., size=(10, 32, 32, 3))
        image_t = tf.constant(image, dtype=tf.float32)
        image_t_discretized = discretization_utils.discretize_uniform(
            image_t, 10)
        # Check counts for quantizing using regular buckets
        counts = tf.reduce_sum(image_t_discretized, axis=-1)
        self.assertTrue(np.all(self.sess.run(counts) == 3))


if __name__ == '__main__':
    unittest.main()
