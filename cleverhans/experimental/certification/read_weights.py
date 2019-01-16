"""File to read in the weights of the neural network from a checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import tensorflow as tf

STRIDE = 2

def read_weights(checkpoint, model_json, input_shape=None, CONV2FF=False):
  """Function to read the weights from checkpoint based on json description.

  Args:
    checkpoint: tensorflow checkpoint with trained model to verify
    model_json: path of json file with model description of the network list
      of dictionary items for each layer containing 'type', 'weight_var',
      'bias_var' and 'is_transpose' 'type'is one of {'ff', 'ff_relu' or
      'conv'}; 'weight_var' is the name of tf variable for weights of layer i;
      'bias_var' is the name of tf variable for bias of layer i;
      'is_transpose' is set to True if the weights have to be transposed as
       per convention
      Note that last layer is always feedforward (verification operates at the
      layer below the final softmax for more numerical stability)
    input_shape: [num_rows, num_columns, num_channels] of the input image
    CONV2FF: Flag to set to True if convolution layer has to be unrolled

  Returns:
    net_weights:
    net_biases:
    net_layer_types: all are lists containing numpy
    version of weights

  Raises:
    ValueError: If layer_types are invalid or variable names not found in
    checkpoint
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
    layer_weight = reader.get_tensor(layer_model_var['weight_var'])
    layer_bias = reader.get_tensor(layer_model_var['bias_var'])
    # TODO(aditirag): is there a way to automatically check when to transpose
    # We want weights W such that x^{i+1} = W^i x^i + b^ix`
    # Can think of a hack involving matching shapes but if shapes are equal
    # it can be ambiguous
    if layer_model_var['type'] in {'ff', 'ff_relu'}:
      layer_weight = np.transpose(layer_weight)
    if layer_model_var['type'] in {'conv'} and CONV2FF:
      layer_model_var['type'] = 'ff'
      current_num_filters = np.shape(layer_weight)[3]
      layer_weight = conv2ff(input_shape, layer_weight)
      current_num_rows = int(input_shape[0] / STRIDE)
      current_num_cols = int(input_shape[1] / STRIDE)
      # Updating input shape for later convolution layers
      input_shape = [current_num_rows, current_num_cols, current_num_filters]
      layer_bias = np.tile(np.reshape(layer_bias, [-1, 1]), 
        [current_num_cols * current_num_rows, 1])
    net_layer_types.append(layer_model_var['type'])
    net_weights.append(layer_weight)
    net_biases.append(np.reshape(layer_bias, (np.size(layer_bias), 1)))
  return net_weights, net_biases, net_layer_types
