from abc import ABCMeta


class Model(object):
    """
    An abstract interface for model wrappers that exposes model symbols
    needed for making an attack. This abstraction removes the dependency on
    any specific neural network package (e.g. Keras) from the core
    code of CleverHans. It can also simplify exposing the hidden features of a
    model when a specific package does not directly expose them.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        # The following is a cache to prevent the re-construction of identical
        # graphs after multiple calls of the fprop methods. The cache is
        # implemented as a dictionary of the form (input, train): output_dict
        # The key is a pair of input (the symbolic representation of the input)
        # and state (a string indicating whether the model is in training,
        # inference, or some other mode --- which may change the behavior of
        # layers like dropout).
        # The output_dict is also a dictionary mapping layer names to symbolic
        # representation of the output of that layer.
        self.fprop_cache = {}

        # By default, we assume the model is being used for training (i.e.,
        # 'train' time). If the model is used for inference or in a different
        # state, a call to set_state() should be made first.
        self._state = 'train'

        pass

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state_val):
        self._state = state_val

    def __call__(self, *args, **kwargs):
        """
        For compatibility with functions used as model definitions (taking
        an input tensor and returning the tensor giving the output
        of the model on that input).
        """
        return self.get_probs(*args, **kwargs)

    def get_layer(self, x, layer):
        """
        Expose the hidden features of a model given a layer name.
        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation of the hidden features
        """
        # Return the symbolic representation for this layer.
        return self.fprop(x)[layer]

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        return self.get_layer(x, 'logits')

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities (i.e.,
                the output values produced by the softmax layer).
        """
        if 'probs' in self.get_layer_names():
            return self.get_layer(x, 'probs')
        else:
            import tensorflow as tf
            return tf.nn.softmax(self.get_logits(x))

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model abstraction.
        """
        raise NotImplementedError('`get_layer_names` not implemented.')

    def fprop(self, x):
        """
        Exposes all the layers of the model returned by get_layer_names.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        # In case of cache hit, return cached dictionary of output tensors.
        if (x, self.state) in self.fprop_cache.keys():
            return self.fprop_cache[(x, self.state)]
        else:
            result = self._fprop(x)
            self.fprop_cache[(x, self.state)] = result
            return result

    def _fprop(self, x):
        raise NotImplementedError('`_fprop` not implemented.')
