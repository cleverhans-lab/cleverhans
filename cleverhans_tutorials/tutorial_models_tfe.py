"""
A TensorFlow Eager implementation of a neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from cleverhans.model import Model


class ModelBasicCNNTFE(Model):
  """
  Basic CNN model for tensorflow eager execution.
  """

  def __init__(self, nb_classes=10,
               nb_filters=64, dummy_input=tf.zeros((32, 28, 28, 1))):
    Model.__init__(self, nb_classes=nb_classes)

    # Parametes
    # number of filters, number of classes.
    self.nb_filters = nb_filters
    self.nb_classes = nb_classes

    # Lists for layers attributes.
    # layer names , layers, layer activations
    self.layer_names = ['input', 'conv_1', 'conv_2', 'conv_3', 'flatten',
                        'logits']
    self.layers = {}
    self.layer_acts = {}

    # layer definitions
    self.layers['conv_1'] = tf.layers.Conv2D(filters=self.nb_filters,
                                             kernel_size=8, strides=2,
                                             padding='same',
                                             activation=tf.nn.relu)
    self.layers['conv_2'] = tf.layers.Conv2D(filters=self.nb_filters * 2,
                                             kernel_size=6, strides=2,
                                             padding='valid',
                                             activation=tf.nn.relu)
    self.layers['conv_3'] = tf.layers.Conv2D(filters=self.nb_filters * 2,
                                             kernel_size=5, strides=1,
                                             padding='valid',
                                             activation=tf.nn.relu)
    self.layers['flatten'] = tf.layers.Flatten()
    self.layers['logits'] = tf.layers.Dense(self.nb_classes,
                                            activation=None)

    # Dummy fprop to activate the network.
    self.fprop(dummy_input)

  def fprop(self, x):
    """
    Forward propagation throught the network
    :return: dictionary with layer names mapping to activation values.
    """

    # Feed forward through the network layers
    for layer_name in self.layer_names:
      if layer_name == 'input':
        prev_layer_act = x
        continue
      else:
        self.layer_acts[layer_name] = self.layers[layer_name](
            prev_layer_act)
        prev_layer_act = self.layer_acts[layer_name]

    # Adding softmax values to list of activations.
    self.layer_acts['probs'] = tf.nn.softmax(
        logits=self.layer_acts['logits'])
    return self.layer_acts

  def get_layer_params(self, layer_name):
    """
    Provides access to the parameters of the given layer.
    Works arounds the non-availability of graph collections in
                eager mode.
    :layer_name: name of the layer for which parameters are
                required, must be one of the string in the
                list layer_names
    :return: list of parameters corresponding to the given
                layer.
    """
    assert layer_name in self.layer_names

    out = []
    layer = self.layers[layer_name]
    layer_variables = layer.variables

    # For each parameter in a layer.
    for param in layer_variables:
      if param not in out:
        out.append(param)
    return out

  def get_params(self):
    """
    Provides access to the model's parameters.
    Works arounds the non-availability of graph collections in
                    eager mode.
    :return: A list of all Variables defining the model parameters.
    """
    assert tf.executing_eagerly()
    out = []

    # Collecting params from each layer.
    for layer_name in self.layers:
      out += self.get_layer_params(layer_name)
    return out

  def get_layer_names(self):
    """:return: the list of exposed layers for this model."""
    return self.layer_names
