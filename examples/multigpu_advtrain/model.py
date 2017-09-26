"""Lightweight model objects in TensorFlow.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import ordered_union

from cleverhans.model import Model

from tensorflow.python.training import moving_averages


def clone_variable(name, x, trainable=False):
    return tf.get_variable(name, shape=x.shape, dtype=x.dtype,
                           trainable=trainable)


class MLP(object):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape, name=None, **kwargs):
        self.layers = layers
        self.input_shape = input_shape
        self.name = name
        if self.name is None:
            self.name = 'MLP'
        self.kwargs = kwargs

    def get_params(self):
        out = []
        for layer in self.layers:
            out = ordered_union(out, layer.get_params())
        return out

    def fprop(self, x, return_all=False, **kwargs):
        kwargs.update(self.kwargs)
        with tf.variable_scope(self.name):
            states = []
            for layer in self.layers:
                x = layer.fprop(x, **kwargs)
                assert x is not None
                states.append(x)
        if return_all:
            return states
        return x

    def set_device(self, device_name):
        for layer in self.layers:
            layer.device_name = device_name

    def __call__(self, x):
        return self.fprop(x)

    def create_sync_ops(self, host_device):
        sync_ops = []
        for layer in self.layers:
            sync_ops += layer.create_sync_ops(host_device)
        return sync_ops


class MLP_probs(Model):
    def __init__(self, mlp, name=None):
        self.mlp = mlp

    def get_layer_names(self):
        return ['probs']

    def set_device(self, device_name):
        self.mlp.set_device(device_name)

    def get_probs(self, x, **kwargs):
        return self.mlp.fprop(x, **kwargs)

    def fprop(self, x, **kwargs):
        probs = self.mlp.fprop(x, **kwargs)
        return {'probs': probs}

    def create_sync_ops(self, host_device):
        return self.mlp.create_sync_ops(host_device)


class Layer(object):
    def __init__(self, input_shape=None, name=None):
        self.input_shape = input_shape
        self.name = name
        self.params_device = {}
        self.params_names = None
        # self.device_name = None  # '/gpu:0'
        self.device_name = '/gpu:0'
        self.training = True

    def get_variable(self, name, initializer):
        v = tf.get_variable(name, shape=initializer.shape,
                            initializer=lambda shape, dtype,
                            partition_info:
                            initializer, trainable=self.training)
        return v

    def set_input_shape_ngpu(self, new_input_shape, **kwargs):
        assert self.device_name
        device_name = self.device_name
        if self.input_shape is None:
            # First time setting the input shape
            self.input_shape = [None] + [int(d) for d in list(new_input_shape)]

        if device_name in self.params_device:
            # There is a copy of weights on this device
            self.__dict__.update(self.params_device[device_name])
            return

        # stop recursion
        self.params_device[device_name] = {}

        # Initialize weights on this device
        with tf.device(device_name):
            keys_before = self.__dict__.keys()
            self.set_input_shape(self.input_shape, **kwargs)
            keys_after = self.__dict__.keys()
            if self.params_names is None:
                self.params_names = list(set(keys_after) - set(keys_before))
            params = dict([(k, self.__dict__[k]) for k in self.params_names])
            self.params_device[device_name] = params

    def create_sync_ops(self, host_device):
        sync_ops = []
        host_params = self.params_device[host_device]
        for device, params in (self.params_device).iteritems():
            if device == host_device:
                continue
            for k in self.params_names:
                if isinstance(params[k], tf.Variable):
                    sync_ops += [tf.assign(params[k], host_params[k])]
        return sync_ops

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x, training=False, **kwargs):
        self.training = training
        if self.name is None:
            self.set_input_shape_ngpu(x.shape[1:], **kwargs)
            return self.fprop_noscope(x, **kwargs)
        else:
            with tf.variable_scope(self.name):
                self.set_input_shape_ngpu(x.shape[1:], **kwargs)
                return self.fprop_noscope(x, **kwargs)


class Linear(Layer):

    def __init__(self, num_hid, w_name='W', **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.num_hid = num_hid
        self.input_shape = None
        self.w_name = w_name

    def set_input_shape(self, input_shape, **kwargs):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        with tf.variable_scope(self.__class__.__name__):
            shape = [dim, self.num_hid]
            init = tf.truncated_normal(shape, stddev=0.1)
            self.W = self.get_variable(self.w_name, init)
            self.b = self.get_variable('b', .1 + np.zeros(
                (self.num_hid,)).astype('float32'))

    def fprop_noscope(self, x, **kwargs):
        return tf.matmul(x, self.W) + self.b


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding,
                 w_name='kernels', *args, **kwargs):
        super(Conv2D, self).__init__(*args, **kwargs)
        self.__dict__.update(locals())
        del self.self
        self.input_shape = None
        self.w_name = w_name

    def set_input_shape(self, input_shape, **kwargs):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        with tf.variable_scope(self.__class__.__name__):
            init = tf.truncated_normal(kernel_shape, stddev=0.1)
            self.kernels = self.get_variable(self.w_name, init)
            self.b = self.get_variable(
                'b', .1 + np.zeros((self.output_channels,)).astype('float32'))
        input_shape = list(input_shape)
        self.input_shape = input_shape
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop_noscope(self, x, **kwargs):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) +
                            (1,), self.padding) + self.b


class ReLU(Layer):

    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)

    def set_input_shape(self, shape, **kwargs):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop_noscope(self, x, **kwargs):
        return tf.nn.relu(x)


class Softmax(Layer):

    def __init__(self, **kwargs):
        super(Softmax, self).__init__(**kwargs)

    def set_input_shape(self, shape, **kwargs):
        self.input_shape = shape
        self.output_shape = shape

    def fprop_noscope(self, x, **kwargs):
        return tf.nn.softmax(x)


class Flatten(Layer):

    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def set_input_shape(self, shape, **kwargs):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop_noscope(self, x, **kwargs):
        return tf.reshape(x, [-1, self.output_width])


class MaxPool(Layer):
    def __init__(self, ksize, strides, padding, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape, **kwargs):
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop_noscope(self, x, **kwargs):
        return tf.nn.max_pool(x,
                              ksize=(1,) + tuple(self.ksize) + (1,),
                              strides=(1,) + tuple(self.strides) + (1,),
                              padding=self.padding)


class BatchNorm(Layer):
    def __init__(self, bn_mean_only=False, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.bn_mean_only = bn_mean_only
        self._extra_train_ops = []
        self.bn_training = True
        self.done_init_training = False

    def set_input_shape(self, input_shape, **kwargs):
        self.input_shape = list(input_shape)
        params_shape = [input_shape[-1]]
        self.params_shape = params_shape

        self.beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=self.training)
        self.gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=self.training)
        self.moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        self.moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        if self.bn_mean_only:
            self.variance = tf.constant(1, tf.float32, params_shape, 'v1')

    def fprop_noscope(self, x, bn_training=False, **kwargs):
        if self.training and bn_training:
            assert not self.done_init_training
            self.done_init_training = True
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            if self.bn_mean_only:
                variance = self.variance

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
            self._extra_train_ops.append(
                moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
        else:
            mean, variance = self.moving_mean, self.moving_variance
        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper
        # net.
        y = tf.nn.batch_normalization(
            x, mean, variance, self.beta, self.gamma, 0.001)
        y.set_shape(x.get_shape())
        return y
