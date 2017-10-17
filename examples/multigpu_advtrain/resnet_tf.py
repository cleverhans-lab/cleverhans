# https://github.com/tensorflow/models/blob/master/research/resnet/resnet_model.py
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import tensorflow as tf
import six

from cleverhans.model import Model

from model import Conv2DnGPU
from model import LinearnGPU
from model import LayerNorm

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer, momentum')


class ResNetTF(Model):
    """ResNet model."""

    def __init__(self, batch_size=None, name=None, optimizer='mom',
                 *args, **kwargs):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.hps = HParams(batch_size=batch_size,
                           num_classes=10,
                           min_lrn_rate=0.0001,
                           lrn_rate=0.1,
                           num_residual_units=5,
                           # num_residual_units=4,
                           use_bottleneck=False,
                           weight_decay_rate=0.0002,
                           relu_leakiness=0.1,
                           optimizer=optimizer,
                           momentum=.9)
        self.reuse = None
        self.kwargs = {}
        self.init_kwargs = kwargs
        del self.init_kwargs['input_shape']
        self.layers = []
        self.layer_idx = 0
        self.init_layers = True
        self.decay_cost = None
        self.training = None
        self.bn_training = None
        self.device_name = None

        self._extra_train_ops = []

    def get_layer_names(self):
        return ['logits', 'probs']

    def set_device(self, device_name):
        self.device_name = device_name
        for layer in self.layers:
            layer.device_name = device_name

    def set_training(self, training, bn_training=False):
        self.training = training
        self.bn_training = bn_training

    def fprop(self, x, return_all=False, dataset='cifar10', **kwargs):
        self.kwargs = kwargs
        self.kwargs.update(self.init_kwargs)
        if 'input_shape' in self.kwargs:
            del self.kwargs['input_shape']
        self.layer_idx = 0
        with tf.variable_scope('Resnet', reuse=self.reuse):
            logits, probs = self._build_model(x)
        self.reuse = True
        states = {'logits': logits, 'probs': probs}
        return states

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self, x):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            # x = self._images
            x = self._conv('init_conv', x, 3, x.shape[3], 16,
                           self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            res_func = self._bottleneck_residual
            filters = [16, 64, 128, 256]
        else:
            res_func = self._residual
            filters = [16, 16, 32, 64]
            # Uncomment the following codes to use w28-10 wide residual
            # network.
            # It is more memory efficient than very deep residual network and
            # has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # filters = [16, 160, 320, 640]
            # Update hps.num_residual_units to 4

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1],
                         self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1],
                             self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2],
                         self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2],
                             self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3],
                         self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3],
                             self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        return logits, self.predictions

    def create_sync_ops(self, host_device):
        sync_ops = []
        for layer in self.layers:
            sync_ops += layer.create_sync_ops(host_device)
        return sync_ops

    def build_cost(self, labels, logits):
        op = logits.op
        if "softmax" in str(op).lower():
            logits, = op.inputs

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            cost = tf.reduce_mean(xent, name='xent')
            cost += self._decay()
            cost = cost

        return cost

    def build_train_op_from_cost(self, cost):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32,
                                    name='learning_rate')
        self.momentum = tf.constant(self.hps.momentum, tf.float32,
                                    name='momentum')

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate,
                                                   self.momentum)

        # there should be no gradients wrt vars on other gpus
        gv_pairs = zip(grads, trainable_variables)
        gv_pairs = [gv for gv in gv_pairs if gv[0] is not None]
        devs = set([gv[1].device for gv in gv_pairs])
        assert len(devs) == 1

        apply_op = optimizer.apply_gradients(
            gv_pairs,
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        train_op = tf.group(*train_ops)
        return train_op

    def _build_train_op(self, logits, labels):
        self.cost = self.build_cost(labels, logits)

        train_op = self.build_train_op_from_cost(self.cost)

        return train_op

    def _batch_norm(self, name, x):
        """Batch normalization."""
        if self.init_layers:
            bn = LayerNorm()
            bn.name = name
            self.layers += [bn]
        else:
            bn = self.layers[self.layer_idx]
            self.layer_idx += 1
        bn.set_training(self.training, self.bn_training)
        x = bn.fprop(x)
        if self.training:
            self._extra_train_ops += bn._extra_train_ops
        return x

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck residual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter / 4,
                           out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv('conv3', x, 1, out_filter /
                           4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1,
                                    in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        if self.decay_cost is not None:
            return self.decay_cost

        costs = []
        if self.device_name is None:
            for var in tf.trainable_variables():
                if var.op.name.find(r'DW') > 0:
                    costs.append(tf.nn.l2_loss(var))
        else:
            for layer in self.layers:
                for var in layer.params_device[self.device_name].values():
                    if (isinstance(var, tf.Variable) and
                            var.op.name.find(r'DW') > 0):
                        costs.append(tf.nn.l2_loss(var))

        self.decay_cost = tf.multiply(self.hps.weight_decay_rate,
                                      tf.add_n(costs))
        return self.decay_cost

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            if self.init_layers:
                conv = Conv2DnGPU(out_filters,
                                  (filter_size, filter_size),
                                  strides[1:3], 'SAME', name=name, w_name='DW')
                conv.name = name
                self.layers += [conv]
            else:
                conv = self.layers[self.layer_idx]
                self.layer_idx += 1
            conv.set_training(self.training)
            return conv.fprop(x)

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x,
                        name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        if self.init_layers:
            fc = LinearnGPU(out_dim, w_name='DW')
            fc.name = 'logits'
            self.layers += [fc]
        else:
            fc = self.layers[self.layer_idx]
            self.layer_idx += 1
        fc.set_training(self.training)
        return fc.fprop(x)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
