"""Code for forward pass through layers of the network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class NeuralNetParams(object):
  """NeuralNetParams is a class that interfaces the verification code with
  the neural net parameters (weights)"""

  def __init__(self, net_weights, net_biases, net_layer_types):
    """Function to initialize NeuralNetParams class.

    Args:
       net_weights: list of numpy matrices of weights of each layer
       [convention: x[i+1] = W[i] x[i]
       net_biases: list of numpy arrays of biases of each layer
       net_layer_types: type of each layer ['ff' or 'ff_relu' or 'ff_conv'
         or 'ff_conv_relu']
         'ff': Simple feedforward layer with no activations
         'ff_relu': Simple feedforward layer with ReLU activations
         'ff_conv': Convolution layer with no activation
         'ff_conv_relu': Convolution layer with ReLU activation

    Raises:
      ValueError: the input lists of net params are not of the same length
    """
    if ((len(net_weights) != len(net_biases))
        or len(net_biases) != len(net_layer_types)):
      raise ValueError('Inputs of net params are not of same length ....')
    if net_layer_types[len(net_layer_types)-1] != 'ff':
      raise ValueError('Final layer is not linear')
    self.num_hidden_layers = len(net_weights) - 1
    self.weights = []
    self.biases = []
    self.layer_types = []
    self.sizes = []
    # Setting the sizes of the layers of the network
    # sizes[i] contains the size of x_i
    for i in range(self.num_hidden_layers):
      shape = np.shape(net_weights[i])
      self.sizes.append(int(shape[1]))
      self.weights.append(
          tf.convert_to_tensor(net_weights[i], dtype=tf.float32))
      self.biases.append(tf.convert_to_tensor(net_biases[i], dtype=tf.float32))
      self.layer_types.append(net_layer_types[i])

    # Last layer shape
    self.sizes.append(int(np.shape(net_weights[self.num_hidden_layers-1])[0]))
    self.final_weights = tf.convert_to_tensor(
        net_weights[self.num_hidden_layers], dtype=tf.float32)
    self.final_bias = tf.convert_to_tensor(
        net_biases[self.num_hidden_layers], dtype=tf.float32)

  def forward_pass(self, vector, layer_index, is_transpose=False, is_abs=False):
    """Performs forward pass through the layer weights at layer_index.

    Args:
      vector: vector that has to be passed through in forward pass
      layer_index: index of the layer
      is_transpose: whether the weights of the layer have to be transposed
      is_abs: whether to take the absolute value of the weights

    Returns:
      tensor that corresponds to the forward pass through the layer

    Raises:
      ValueError: if the layer_index is negative or more than num hidden layers
    """
    if(layer_index < 0 or layer_index > self.num_hidden_layers):
      raise ValueError('Invalid layer index')

    if(self.layer_types[layer_index] == 'ff' or
       self.layer_types[layer_index] == 'ff_relu'):
      if is_transpose:
        return tf.matmul(tf.transpose(self.weights[layer_index]), vector)
      elif is_abs:
        return tf.matmul(tf.abs(self.weights[layer_index]), vector)
      else:
        return tf.matmul(self.weights[layer_index], vector)

    raise NotImplementedError('Unsupported layer type: {0}'.format(
        self.layer_types[layer_index]))
