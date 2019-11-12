"""Extremely simple model where all parameters are from convolutions.
"""

import math
import tensorflow as tf
import numpy as np
from cleverhans import initializers
from cleverhans.serial import NoRefModel

class Layer(object):
  def get_output_shape(self):
    return self.output_shape

class Conv2D(Layer):
  """
  A simple model that uses only convolution and downsampling---no batch norm or other techniques
  """
  def __init__(self, scope, nb_filters):   
    self.nb_filters = nb_filters
    self.scope = scope
    
  def set_input_shape(self, input_shape):
    batch_size, rows, cols, input_channels = input_shape
    input_shape = list(input_shape)
    input_shape[0] = 1
    dummy_batch = tf.zeros(input_shape)
    dummy_output = self.fprop(dummy_batch)
    output_shape = [int(e) for e in dummy_output.get_shape()]
    output_shape[0] = batch_size
    self.output_shape = tuple(output_shape)
    
  def fprop(self, x):
    conv_args = dict(
        activation=tf.nn.leaky_relu,
        kernel_initializer=initializers.HeReLuNormalInitializer,
        kernel_size=3,
        padding='same')
  

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      for scale in range(3):
        x = tf.layers.conv2d(x, self.nb_filters << scale, **conv_args)
        x = tf.layers.conv2d(x, self.nb_filters << (scale + 1), **conv_args)
        x = tf.layers.average_pooling2d(x, 2, 2)
      
      # reshape the output of conv to be the input of capsule
      num_capsules = x.get_shape().as_list()[1] * x.get_shape().as_list()[2]
      input_atoms = x.get_shape().as_list()[3]
      x =  tf.reshape(x, [-1, num_capsules, input_atoms])
      return x

class Capsule(Layer):
  """ Capsule layer
      input dim: batch_size, num_capsules_input, input_atoms
      output dim: batch_size, num_capsules_output, output_atoms
  """

  def __init__(self, scope, num_capsules_output, output_atoms, num_routing):
    self.scope = scope
    self.num_capsules_output = num_capsules_output
    self.output_atoms = output_atoms
    self.num_routing = num_routing

  def set_input_shape(self, input_shape):
    batch_size, num_capsules_input, input_atoms  = input_shape
    self.num_capsules_input = num_capsules_input
    self.input_atoms = input_atoms
    self.output_shape = [batch_size, self.num_capsules_output, self.output_atoms]
    self.make_vars()
    
  def make_vars(self):
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      w = tf.get_variable('DW', [self.num_capsules_input, self.input_atoms, self.num_capsules_output * self.output_atoms], initializer=tf.initializers.truncated_normal(stddev=0.03))
      b = tf.get_variable('bias', [self.num_capsules_output, self.output_atoms], initializer=tf.initializers.constant())
    return w, b

  def _squash(self, input_tensor):
    """Applies norm nonlinearity (squash) to a capsule layer.
    Args:
      input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
        fully connected capsule layer or
        [batch, num_channels, num_atoms, height, width] for a convolutional
        capsule layer.
    Returns:
      A tensor with same shape as input (rank 3) for output of this layer.
    """
    
    norm = tf.norm(input_tensor, axis=2, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


  def _leaky_routing(self, logits, output_dim):
      
    leak = tf.zeros_like(logits, optimize=True)
    leak = tf.reduce_sum(leak, axis=2, keep_dims=True)
    leaky_logits = tf.concat([leak, logits], axis=2)
    leaky_routing = tf.nn.softmax(leaky_logits, dim=2)
    return tf.split(leaky_routing, [1, output_dim], 2)[1]


  def _update_routing(self, votes, biases, logit_shape, num_dims, input_dim, output_dim,
                        num_routing, leaky):
    votes_t_shape = [3, 0, 1, 2]
    for i in range(num_dims - 4):
      votes_t_shape += [i + 4]
    r_t_shape = [1, 2, 3, 0]
    for i in range(num_dims - 4):
      r_t_shape += [i + 4]
    votes_trans = tf.transpose(votes, votes_t_shape)

    def _body(i, logits, activations):
      """Routing while loop."""
      # route: [batch, input_dim, output_dim, ...]
      if leaky:
        route = self._leaky_routing(logits, output_dim)
      else:
        route = tf.nn.softmax(logits, dim=2)
      preactivate_unrolled = route * votes_trans
      #route.shape (16,?, 49, 32)
      #votes_trans.shape (16, ?, 49, 32)
      preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
      preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
      activation = self._squash(preactivate)
      activations = activations.write(i, activation)
      # distances: [batch, input_dim, output_dim]
      act_3d = tf.expand_dims(activation, 1)
      tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
      tile_shape[1] = input_dim
      act_replicated = tf.tile(act_3d, tile_shape)
      distances = tf.reduce_sum(votes * act_replicated, axis=3)
      logits += distances
      return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)

    return activations.read(num_routing - 1)


  def fprop(self, x):
    for i in range(1):
       
        with tf.name_scope(self.scope):
          weights, biases = self.make_vars()
          input_tiled = tf.tile(
              tf.expand_dims(x, -1),
              [1, 1, 1, self.num_capsules_output * self.output_atoms])
          votes = tf.reduce_sum(input_tiled * weights, axis=2)
          votes_reshaped = tf.reshape(votes,
                                      [-1, self.num_capsules_input, self.num_capsules_output, self.output_atoms])
        
          input_shape = tf.shape(x)
          logit_shape = tf.stack([input_shape[0], self.num_capsules_input, self.num_capsules_output])
          activations = self._update_routing(
              votes=votes_reshaped,
              biases=biases,
              logit_shape=logit_shape,
              num_dims=4,
              input_dim=self.num_capsules_input,
              output_dim=self.num_capsules_output,
              num_routing=self.num_routing,
              leaky=True)
    return activations

class Reconstruction(Layer):
    ''' Reconstruction Network:
        return: a concatenation of nb_classes logits and the winning-capsule recontruction
                shape: (batch_size, nb_classes + 1d image shape)

    '''
    def __init__(self, scope, nb_classes):
        self.scope = scope
        self.nb_classes = nb_classes
        
    
    def set_input_shape(self, input_shape):
        self.batch_size, _, self.num_atoms = input_shape
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = dummy_output.get_shape().as_list()
        output_shape[0] = self.batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x, **kwargs):
        # the first num_classes capsules are used for classification
        logit =  tf.norm(x[:, :self.nb_classes, :], axis=-1)
        
        # use the predicted label to construct the mask
        mask = tf.one_hot(tf.argmax(logit, axis=-1), self.nb_classes)
        bg = tf.ones_like(x[:, self.nb_classes:, 0])
        mask_bg = tf.concat([mask, bg], axis=-1)
        capsule_mask_3d = tf.expand_dims(mask_bg, -1)
        atom_mask = tf.tile(capsule_mask_3d, [1, 1, self.num_atoms])
        filtered_embedding = x * atom_mask
        filtered_embedding_2d = tf.contrib.layers.flatten(filtered_embedding)

        # feed the extracted class capsule + background capsules into the reconstruction network
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
                net = tf.contrib.layers.fully_connected(filtered_embedding_2d, 1024)
                net = tf.contrib.layers.fully_connected(net, 32//4 * 32//4 * 256)
                net = tf.reshape(net, [-1, 32//4, 32//4, 256])
                net = tf.contrib.layers.conv2d_transpose(net, 64, [4, 4], stride=2)
                net = tf.contrib.layers.conv2d_transpose(net, 32, [4, 4], stride=2)
                net = tf.layers.conv2d(net, 3, kernel_size=4, padding='same')
                net = tf.sigmoid(net)              
                reconstruction_2d = tf.layers.flatten(net)
        return tf.concat([logit, reconstruction_2d], axis=-1)
        
# extract the class logits for classification
class IdentityRecons(Layer):
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
    
    def set_input_shape(self, input_shape):
        batch_size, _  = input_shape
        self.output_shape = [batch_size, self.nb_classes]
        
    def fprop(self, x):
        return x[:, :self.nb_classes]
    
class Network(NoRefModel):
  """CapsNet model."""

  def __init__(self, layers, input_shape, nb_classes, scope=None):
    """
    :param layers: a list of layers in CleverHans format
      each with set_input_shape() and fprop() methods.
    :param input_shape: 4-tuple describing input shape (e.g None, 32, 32, 3)
    :param scope: string name of scope for Variables
    """
    super(Network, self).__init__(scope, nb_classes, {}, scope is not None)   
    with tf.variable_scope(self.scope):
      self.build(layers, input_shape, nb_classes)

  def get_vars(self):
    if hasattr(self, "vars"):
      return self.vars
    return super(Network, self).get_vars()

  def build(self, layers, input_shape, nb_classes):
      self.layer_names = []
      self.layers = layers
      self.input_shape = input_shape
      self.nb_classes = nb_classes
      layers[-2].name = 'recons' 
      layers[-1].name = 'logits'
      for i, layer in enumerate(self.layers):
        if hasattr(layer, 'name'):
          name = layer.name
        else:
          name = layer.__class__.__name__ + str(i)
          layer.name = name
        self.layer_names.append(name)

        layer.set_input_shape(input_shape)
        input_shape = layer.get_output_shape()

  def make_input_placeholder(self):
    return tf.placeholder(tf.float32, (None, self.input_shape[1], self.input_shape[2], self.input_shape[3]))

  def make_label_placeholder(self):
    return tf.placeholder(tf.float32, (None, self.nb_classes))

  def fprop(self, x, set_ref=False, **kwargs):
    if self.scope is not None:
      with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
        return self._fprop(x, set_ref, **kwargs)
    return self._prop(x, set_ref)

  def _fprop(self, x, set_ref=False, **kwargs):
      states = []
      for layer in self.layers:
        if set_ref:
          layer.ref = x
        else:
            x = layer.fprop(x)
        assert x is not None
        states.append(x)
      states = dict(zip(self.layer_names, states))
      return states

  def add_internal_summaries(self):
    pass


# Convolutional Layers + CapsLayer + Reconstruction      
def CapsNetRecons(scope, nb_classes, nb_filters, input_shape, num_capsules_output, output_atoms, num_routing):
    layers=[Conv2D(scope, nb_filters),
            Capsule("CapsLayer", num_capsules_output, output_atoms, num_routing),
            Reconstruction("ReconsLayer", nb_classes),
            IdentityRecons(nb_classes)]
    model = Network(layers, input_shape, nb_classes, scope)
    return model



