"""This file defines the neural network class, where a network is reinitialized from configuration files.

The class also has a forward propagation method.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import tensorflow as tf

class NeuralNetwork(object):
  """NeuralNetwork is a class that interfaces the verification code with
  the neural net parameters (weights).
  """

  def __init__(self, net_weights, net_biases, net_layer_types):
    """Function to initialize NeuralNetParams class.

    Args:
      net_weights: list of numpy matrices of weights of each layer
       [convention: x[i+1] = W[i] x[i]
      net_biases: list of numpy arrays of biases of each layer
      net_layer_types: type of each layer ['ff' or 'ff_relu' or 'ff_conv' or
        'ff_conv_relu']
        'ff': Simple feedforward layer with no activations
        'ff_relu': Simple feedforward layer with ReLU activations
        'ff_conv': Convolution layer with no activation
        'ff_conv_relu': Convolution layer with ReLU activation

    Raises:
      ValueError: the input lists of net params are not of the same length
    """
    if ((len(net_weights) != len(net_biases)) or
        len(net_biases) != len(net_layer_types)):
      raise ValueError('Inputs of net params are not of same length ....')
    if net_layer_types[len(net_layer_types) - 1] != 'ff':
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
    self.sizes.append(int(np.shape(net_weights[self.num_hidden_layers - 1])[0]))
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
    if (layer_index < 0 or layer_index > self.num_hidden_layers):
      raise ValueError('Invalid layer index')

    if (self.layer_types[layer_index] == 'ff' or
        self.layer_types[layer_index] == 'ff_relu'):
      if is_transpose:
        return tf.matmul(tf.transpose(self.weights[layer_index]), vector)
      elif is_abs:
        return tf.matmul(tf.abs(self.weights[layer_index]), vector)
      else:
        return tf.matmul(self.weights[layer_index], vector)

    raise NotImplementedError('Unsupported layer type: {0}'.format(
        self.layer_types[layer_index]))

def load_network_from_checkpoint(checkpoint, model_json):
  """Function to read the weights from checkpoint based on json description.

    Args:
      checkpoint: tensorflow checkpoint with trained model to
        verify
      model_json: path of json file with model description of
        the network list of dictionary items for each layer
        containing 'type', 'weight_var', 'bias_var' and
        'is_transpose' 'type'is one of {'ff', 'ff_relu' or
        'conv'}; 'weight_var' is the name of tf variable for
        weights of layer i; 'bias_var' is the name of tf
        variable for bias of layer i; 'is_transpose' is set to
        True if the weights have to be transposed as per
        convention Note that last layer is always feedforward
        (verification operates at the layer below the final
        softmax for more numerical stability)

    Returns:
      net_weights: list of numpy matrices of weights of each layer
        convention: x[i+1] = W[i] x[i]
      net_biases: list of numpy arrays of biases of each layer
      net_layer_types: type of each layer ['ff' or 'ff_relu' or 'ff_conv'
        or 'ff_conv_relu']
        'ff': Simple feedforward layer with no activations
        'ff_relu': Simple feedforward layer with ReLU activations
        'ff_conv': Convolution layer with no activation
        'ff_conv_relu': Convolution layer with ReLU activation

    Raises:
      ValueError: If layer_types are invalid or variable names
        not found in checkpoint
  """
  # Load checkpoint
  reader = tf.train.load_checkpoint(checkpoint)
  variable_map = reader.get_variable_to_shape_map()
  checkpoint_variable_names = variable_map.keys()
  # Parse JSON file for names
  with tf.gfile.Open(model_json) as f:
    list_model_var = json.load(f)

  net_layer_types = []
  net_weights = []
  net_biases = []

  # Checking validity of the input and adding to list
  for layer_model_var in list_model_var:
    if layer_model_var['type'] not in {'ff', 'ff_relu', 'conv'}:
      raise ValueError('Invalid layer type in description')
    if (layer_model_var['weight_var'] not in checkpoint_variable_names or
        layer_model_var['bias_var'] not in checkpoint_variable_names):
      raise ValueError('Variable names not found in checkpoint')
    net_layer_types.append(layer_model_var['type'])
    layer_weight = reader.get_tensor(layer_model_var['weight_var'])
    layer_bias = reader.get_tensor(layer_model_var['bias_var'])
    # TODO(aditirag): is there a way to automatically check when to transpose
    # We want weights W such that x^{i+1} = W^i x^i + b^i
    # Can think of a hack involving matching shapes but if shapes are equal
    # it can be ambiguous
    if layer_model_var['is_transpose']:
      layer_weight = np.transpose(layer_weight)
    net_weights.append(layer_weight)
    net_biases.append(np.reshape(layer_bias, (np.size(layer_bias), 1)))
  return NeuralNetwork(net_weights, net_biases, net_layer_types)
