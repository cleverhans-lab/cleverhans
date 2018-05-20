"""cleverhans.model.Model implementation of cifar10_challenge.model.Model

This re-implementation factors variable creation apart from forward
propagation so it is possible to run forward propagation more than once
in the same model.

based on https://github.com/tensorflow/models/tree/master/resnet
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from cleverhans.model import Model


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class ResNet(Model):
    """ResNet model."""

    def __init__(self, layers, input_shape):
        """ResNet constructor.

        Args:
          layers: a list of layers in CleverHans format
          each with set_input_shape() and fprop() methods.
          input_shape: 4-tuple describing input shape (e.g None, 32, 32, 3)
        """
        super(ResNet, self).__init__()
        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states

    def add_internal_summaries(self):
        pass


def _stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]


class Input(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        # assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            input_standardized = tf.map_fn(
                lambda img: tf.image.per_image_standardization(img), x)
            return _conv('init_conv', input_standardized,
                         3, 3, 16, _stride_arr(1))


class Conv2D(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):

        # Update hps.num_residual_units to 9
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        filters = [16, 160, 320, 640]
        res_func = _residual
        with tf.variable_scope('unit_1_0', reuse=tf.AUTO_REUSE):
            x = res_func(x, filters[0], filters[1], _stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, 5):
            with tf.variable_scope(('unit_1_%d' % i), reuse=tf.AUTO_REUSE):
                x = res_func(x, filters[1], filters[1],
                             _stride_arr(1), False)

        with tf.variable_scope(('unit_2_0'), reuse=tf.AUTO_REUSE):
            x = res_func(x, filters[1], filters[2], _stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, 5):
            with tf.variable_scope(('unit_2_%d' % i), reuse=tf.AUTO_REUSE):
                x = res_func(x, filters[2], filters[2],
                             _stride_arr(1), False)

        with tf.variable_scope(('unit_3_0'), reuse=tf.AUTO_REUSE):
            x = res_func(x, filters[2], filters[3], _stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, 5):
            with tf.variable_scope(('unit_3_%d' % i), reuse=tf.AUTO_REUSE):
                x = res_func(x, filters[3], filters[3],
                             _stride_arr(1), False)

        with tf.variable_scope(('unit_last'), reuse=tf.AUTO_REUSE):
            x = _batch_norm('final_bn', x)
            x = _relu(x, 0.1)
            x = _global_avg_pool(x)

        return x


class Linear(Layer):

    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]

        with tf.variable_scope('logit', reuse=tf.AUTO_REUSE):
            self.w = tf.get_variable(
                'DW', [dim, self.num_hid],
                initializer=tf.initializers.variance_scaling(
                    distribution='uniform'))
            self.b = tf.get_variable('biases', [self.num_hid],
                                     initializer=tf.initializers.constant())

    def fprop(self, x):
        return tf.nn.xw_plus_b(x, self.w, self.b)


def _batch_norm(name, x):
    """Batch normalization."""
    with tf.name_scope(name):
        return tf.contrib.layers.batch_norm(
            inputs=x,
            decay=.9,
            center=True,
            scale=True,
            activation_fn=None,
            updates_collections=None,
            is_training=False)


def _residual(x, in_filter, out_filter, stride,
              activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            x = _batch_norm('init_bn', x)
            x = _relu(x, 0.1)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = _batch_norm('init_bn', x)
            x = _relu(x, 0.1)

    with tf.variable_scope('sub1'):
        x = _conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
        x = _batch_norm('bn2', x)
        x = _relu(x, 0.1)
        x = _conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0],
                         [0, 0], [(out_filter - in_filter) // 2,
                                  (out_filter - in_filter) // 2]])
        x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x


def _decay():
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find('DW') > 0:
            costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)


def _conv(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')


def _relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def _fully_connected(x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
        prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.initializers.variance_scaling(distribution='uniform'))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.initializers.constant())
    return tf.nn.xw_plus_b(x, self.w, self.b)


def _global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


class Softmax(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


def make_madry_wresnet(nb_classes=10, input_shape=(None, 32, 32, 3)):
    layers = [Input(),
              Conv2D(),  # the whole ResNet is basically created in this layer
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = ResNet(layers, input_shape)
    return model
