"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

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


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):

    def __init__(self, num_hid, bias_init=0., weight_init=None):
        self.num_hid = num_hid
        self.bias_init = bias_init
        self.weight_init = weight_init

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        shape = [dim, self.num_hid]
        if self.weight_init == 't_normal':
            init = tf.truncated_normal(shape, stddev=0.1)
        else:
            init = tf.random_normal(shape, dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                       keep_dims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding,
                 bias_init=0., weight_init=None):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        if self.weight_init == 't_normal':
            init = tf.truncated_normal(kernel_shape, stddev=0.1)
        else:
            init = tf.random_normal(kernel_shape, dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                       axis=(0, 1, 2)))
        self.kernels = tf.Variable(init)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        # orig_input_batch_size = input_shape[0]
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b


class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.relu(x)


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


class MaxPool(Layer):
    def __init__(self, ksize, strides, padding):
        super(MaxPool, self).__init__()
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.max_pool(x,
                              ksize=(1,) + tuple(self.ksize) + (1,),
                              strides=(1,) + tuple(self.strides) + (1,),
                              padding=self.padding)


def make_basic_cnn(nb_filters=64, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model


def make_madryetal(nb_classes=10, input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(32, (5, 5), (1, 1), "SAME", bias_init=.1,
                     weight_init='t_normal'),
              ReLU(),
              MaxPool((2, 2), (2, 2), "SAME"),
              Conv2D(64, (5, 5), (1, 1), "SAME", bias_init=.1,
                     weight_init='t_normal'),
              ReLU(),
              MaxPool((2, 2), (2, 2), "SAME"),
              Flatten(),
              Linear(1024, bias_init=.1, weight_init='t_normal'),
              ReLU(),
              Linear(nb_classes, bias_init=.1, weight_init='t_normal'),
              Softmax()]

    model = MLP(layers, input_shape)
    return model
