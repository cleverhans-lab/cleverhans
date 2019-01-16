"""Code for forward pass through layers of the network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

STRIDE = 2
PADDING = 'SAME'

class NeuralNetParams(object):
  """NeuralNetParams is a class that interfaces the verification code with
  the neural net parameters (weights)"""

  def __init__(self, net_weights, net_biases, net_layer_types, input_shape=None):
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
       input_shape = [num_rows, num_columns, num_channels] at the input layer

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
    self.input_shapes = []
    self.output_shapes = []
    if(input_shape is not None):
      current_num_rows = input_shape[0]
      current_num_columns = input_shape[1]
      current_num_channels = input_shape[2]

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
        num_filters = shape[3]
        self.input_shapes.append([1, 
          current_num_rows, current_num_cols, current_num_channels])
        self.sizes.append(current_num_rows*current_num_cols*current_num_channels)
        current_num_channels = num_filters
        # For propagating across multiple conv layers
        if(PADDING == 'SAME'):
          current_num_rows = int(current_num_rows/STRIDE)
          current_num_cols = int(current_num_cols/STRIDE)
        self.output_shapes.append(
          [1, current_num_rows, current_num_cols, current_num_channels])

        # For conv networks, unraveling the bias terms                                                                                           
        small_bias = tf.convert_to_tensor(net_biases[i], dtype=tf.float32)
        large_bias = tf.tile(tf.reshape(small_bias, [-1, 1]), 
          [current_num_rows*current_num_cols, 1])
        self.biases.append(large_bias)

    # Last layer shape: always ff 
    final_dim = int(np.shape(net_weights[self.num_hidden_layers])[1])
    self.sizes.append(final_dim)
    self.input_shapes.append([final_dim, 1])
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
    # Reshaping the input for convolution appropriately
    if is_transpose:
      vector = tf.reshape(vector, self.output_shapes[layer_index])
      if (layer_type in {'ff', 'ff_relu'}):
        return_vector = tf.matmul(tf.transpose(self.weights[layer_index]), vector)
      elif(layer_type in {'conv', 'conv_relu'}):
        return_vector = tf.nn.conv2d_transpose(vector, self.weights[layer_index],
                                               output_shape=self.input_shapes[layer_index],
                                               strides=[1, STRIDE, STRIDE, 1],
                                               padding=PADDING)
      else:
        raise NotImplementedError('Unsupported layer type: {0}'.format(self.layer_type))
      return tf.reshape(return_vector, (self.sizes[layer_index + 1], 1))

    elif is_abs:
      vector = tf.reshape(vector, self.input_shapes[layer_index])
      if(layer_type in {'ff', 'ff_relu'}):
        return_vector = tf.matmul(tf.abs(self.weights[layer_index], vector))
      elif(layer_type in {'conv', 'conv_relu'}):
        return_vector = tf.nn.conv2d(vector, 
                                    tf.abs(self.weights[layer_index]), 
                                    strides=[1, STRIDE, STRIDE, 1],
                                    padding=PADDING)
      else:
        raise NotImplementedError('Unsupported layer type: {0}'.format(self.layer_type))
      return tf.reshape(return_vector, (self.sizes[layer_index + 1], 1))

    # Simple forward pass 
    else:
      vector = tf.reshape(vector, self.input_shapes[layer_index])
      if(layer_type in {'ff', 'ff_relu'}):
        return_vector = tf.matmul(self.weights[layer_index], vector)
      elif (layer_type in {'conv', 'conv_relu'}):
        return_vector = tf.nn.conv2d(vector, self.weights[layer_index], 
                                    strides=[1, STRIDE, STRIDE, 1],
                                    padding=PADDING)
      else:
        raise NotImplementedError('Unsupported layer type: {0}'.format(self.layer_type))
      return tf.reshape(return_vector, (self.sizes[layer_index + 1], 1))


    raise NotImplementedError('Unsupported layer type: {0}'.format(
        self.layer_types[layer_index]))

  def nn_output(self, test_input, true_class, adv_class):
    """ Function to print the output of forward pass according the neural net class
      Args:
        test_input: Input to pass through the network
        true_class: True class of the input 
        adv_class: Adversarial class to be considered
    """
    activation = test_input
    # Assumes that all the layers are relu: has to be modified for other architectures
    for i in range(self.num_hidden_layers):
      activation = tf.nn.relu(self.forward_pass(activation, i) + self.biases[i])
    # Final layer                                                                                                                                
    activation = tf.reshape(activation, [-1, 1])
    final_weight = self.final_weights[adv_class, :] - self.final_weights[true_class, :]
    final_weight = tf.reshape(final_weight, [-1, 1])
    final_constant = (self.final_bias[adv_class] - self.final_bias[true_class])
    output = tf.reduce_sum(tf.multiply(activation, final_weight)) + final_constant
    return output
