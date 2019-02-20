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

  def __init__(self, net_weights, net_biases, net_layer_types, input_shape=None, cnn_params=None):
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
      input_shape: [num_rows, num_columns, num_channels] at the input layer
      cnn_params: list of dictionaries containing stride and padding for
        each layer

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
    self.input_shapes = []
    self.output_shapes = []
    self.has_conv = False
    if(input_shape is not None):
      current_num_rows = input_shape[0]
      current_num_columns = input_shape[1]
      current_num_channels = input_shape[2]
    self.cnn_params = cnn_params

    # Setting the sizes of the layers of the network
    # sizes[i] contains the size of x_i
    for i in range(self.num_hidden_layers):
      shape = np.shape(net_weights[i])
      self.weights.append(
          tf.convert_to_tensor(net_weights[i], dtype=tf.float32))
      self.layer_types.append(net_layer_types[i])

      if(self.layer_types[i] in {'ff', 'ff_relu'}):
        self.sizes.append(int(shape[1]))
        # For feedforward networks, no unraveling the bias terms

        small_bias = tf.convert_to_tensor(net_biases[i], dtype=tf.float32)
        self.biases.append(tf.reshape(small_bias, [-1, 1]))
        # Assumes that x^{i+1} = W_i x^i
        self.input_shapes.append([int(shape[1]), 1])
        self.output_shapes.append([int(shape[0]), 1])

      # Convolution type
      else:
        self.has_conv = True
        num_filters = shape[3]
        self.input_shapes.append([1, current_num_rows, current_num_columns, current_num_channels])
        self.sizes.append(current_num_rows*current_num_columns*current_num_channels)
        current_num_channels = num_filters
        # For propagating across multiple conv layers
        if(self.cnn_params[i]['padding'] == 'SAME'):
          current_num_rows = int(current_num_rows/self.cnn_params[i]['stride'])
          current_num_columns = int(current_num_columns/self.cnn_params[i]['stride'])
        self.output_shapes.append(
            [1, current_num_rows, current_num_columns, current_num_channels])

        # For conv networks, unraveling the bias terms
        small_bias = tf.convert_to_tensor(net_biases[i], dtype=tf.float32)
        large_bias = tf.tile(tf.reshape(small_bias, [-1, 1]),
                             [current_num_rows*current_num_columns, 1])
        self.biases.append(large_bias)

    # Last layer shape: always ff
    if self.has_conv:
      final_dim = int(np.shape(net_weights[self.num_hidden_layers])[1])
      self.input_shapes.append([final_dim, 1])

    else:
      final_dim = int(np.shape(net_weights[self.num_hidden_layers - 1])[0])

    self.sizes.append(final_dim)
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

    layer_type = self.layer_types[layer_index]
    weight = self.weights[layer_index]
    if is_abs:
      weight = tf.abs(weight)
    if is_transpose:
      vector = tf.reshape(vector, self.output_shapes[layer_index])
    else:
      vector = tf.reshape(vector, self.input_shapes[layer_index])

    if layer_type in {'ff', 'ff_relu'}:
      if is_transpose:
        weight = tf.transpose(weight)
      return_vector = tf.matmul(weight, vector)
    elif layer_type in {'conv', 'conv_relu'}:
      if is_transpose:
        return_vector = tf.nn.conv2d_transpose(vector,
                                               weight,
                                               output_shape=self.input_shapes[layer_index],
                                               strides=[1, self.cnn_params[layer_index]['stride'],
                                                        self.cnn_params[layer_index]['stride'], 1],
                                               padding=self.cnn_params[layer_index]['padding'])
      else:
        return_vector = tf.nn.conv2d(vector,
                                     weight,
                                     strides=[1, self.cnn_params[layer_index]['stride'],
                                              self.cnn_params[layer_index]['stride'], 1],
                                     padding=self.cnn_params[layer_index]['padding'])
    else:
      raise NotImplementedError('Unsupported layer type: {0}'.format(layer_type))
    if is_transpose:
      return tf.reshape(return_vector, (self.sizes[layer_index], 1))
    return tf.reshape(return_vector, (self.sizes[layer_index + 1], 1))

def load_network_from_checkpoint(checkpoint, model_json, input_shape=None):
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
  cnn_params = []

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
    if layer_model_var['type'] in {'ff', 'ff_relu'}:
      layer_weight = np.transpose(layer_weight)
      cnn_params.append(None)
    if layer_model_var['type'] in {'conv'}:
      if 'stride' not in layer_model_var or 'padding' not in layer_model_var:
        raise ValueError('Please define stride and padding for conv layers.')
      cnn_params.append({'stride': layer_model_var['stride'], 'padding': layer_model_var['padding']})
    net_weights.append(layer_weight)
    net_biases.append(np.reshape(layer_bias, (np.size(layer_bias), 1)))
  return NeuralNetwork(net_weights, net_biases, net_layer_types, input_shape, cnn_params)
