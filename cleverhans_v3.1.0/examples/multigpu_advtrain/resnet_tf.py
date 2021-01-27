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

from model import MLPnGPU
from model import Conv2DnGPU
from model import LinearnGPU
from model import LayerNorm

HParams = namedtuple('HParams',
                     'batch_size, nb_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, momentum')


class ResNetTF(MLPnGPU):
  """ResNet model."""

  def __init__(self, batch_size=None, name=None, **kwargs):
    NB_CLASSES = 10
    super(ResNetTF, self).__init__(nb_classes=NB_CLASSES, layers=[],
                                   input_shape=None)
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self.hps = HParams(batch_size=batch_size,
                       nb_classes=NB_CLASSES,
                       min_lrn_rate=0.0001,
                       lrn_rate=0.1,
                       num_residual_units=5,
                       use_bottleneck=False,
                       weight_decay_rate=0.0002,
                       relu_leakiness=0.1,
                       momentum=.9)
    self.layers = []
    self.layer_idx = 0
    self.init_layers = True
    self.decay_cost = None
    self.training = None
    self.device_name = None

  def set_training(self, training=False):
    super(ResNetTF, self).set_training(training)
    self.training = training

  def fprop(self, x):
    self.layer_idx = 0
    with tf.variable_scope('Resnet'):
      logits, probs = self._build_model(x)
    self.init_layers = False
    states = {'logits': logits, 'probs': probs}
    return states

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, x):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
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
      x = self._layer_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.nb_classes)
      predictions = tf.nn.softmax(logits)

    return logits, predictions

  def build_cost(self, labels, logits):
    """
    Build the graph for cost from the logits if logits are provided.
    If predictions are provided, logits are extracted from the operation.
    """
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
    devs = {v.device for v in trainable_variables}
    assert len(devs) == 1, ('There should be no trainable variables'
                            ' on any device other than the last GPU.')

    optimizer = tf.train.MomentumOptimizer(self.lrn_rate, self.momentum)

    gv_pairs = zip(grads, trainable_variables)
    gv_pairs = [gv for gv in gv_pairs if gv[0] is not None]
    devs = {gv[1].device for gv in gv_pairs}
    assert len(devs) == 1, ('There should be no gradients wrt'
                            ' vars on other GPUs.')

    apply_op = optimizer.apply_gradients(
        gv_pairs,
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op]
    train_op = tf.group(*train_ops)
    return train_op

  def _layer_norm(self, name, x):
    """Layer normalization."""
    if self.init_layers:
      bn = LayerNorm()
      bn.name = name
      self.layers += [bn]
    else:
      bn = self.layers[self.layer_idx]
      self.layer_idx += 1
    bn.device_name = self.device_name
    bn.set_training(self.training)
    x = bn.fprop(x)
    return x

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._layer_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._layer_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._layer_norm('bn2', x)
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

    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._layer_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._layer_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

    with tf.variable_scope('sub2'):
      x = self._layer_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter / 4,
                     out_filter / 4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._layer_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter /
                     4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1,
                            in_filter, out_filter, stride)
      x += orig_x

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
          if (isinstance(var, tf.Variable) and var.op.name.find(r'DW') > 0):
            costs.append(tf.nn.l2_loss(var))

    self.decay_cost = tf.multiply(self.hps.weight_decay_rate,
                                  tf.add_n(costs))
    return self.decay_cost

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    if self.init_layers:
      conv = Conv2DnGPU(out_filters,
                        (filter_size, filter_size),
                        strides[1:3], 'SAME', w_name='DW')
      conv.name = name
      self.layers += [conv]
    else:
      conv = self.layers[self.layer_idx]
      self.layer_idx += 1
    conv.device_name = self.device_name
    conv.set_training(self.training)
    return conv.fprop(x)

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    if self.init_layers:
      fc = LinearnGPU(out_dim, w_name='DW')
      fc.name = 'logits'
      self.layers += [fc]
    else:
      fc = self.layers[self.layer_idx]
      self.layer_idx += 1
    fc.device_name = self.device_name
    fc.set_training(self.training)
    return fc.fprop(x)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
