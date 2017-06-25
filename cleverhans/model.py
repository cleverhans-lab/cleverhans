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

    def __init__(self, model):
        """
        If `fprop_layer` is implemented, `__init__`
        should keep track of the name of the layers or `self.model` should
        provide a method for retrieving a layer.
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's post-softmax predictions
                      (probabilities).
        """
        self.model = model

        # The following is a cache to prevent the re-construction of identical
        # graphs after multiple calls of the fprop methods. The cache is
        # implemented as a dictionary of the form (input, train): output_dict
        # The key is a pair of input (the symbolic representation of the input)
        # and train (a boolean indicating whether the graph is in training or
        # inference mode---which changes the behavior of layers like dropout).
        # The values output_dict are also a dictionary mapping layer names
        # to symbolic representation of the output of that layer.
        self.fprop_cache = {}

        # By default, we assume the model is being used for inference (i.e.,
        # 'test' time). If the model is being trained or in another state,
        # a call to set_state() should be made first.
        self.state = 'test'

        pass

    def __call__(self, *args, **kwargs):
        """
        For compatibilty with a standard model.
        """
        return self.fprop_probs(*args, **kwargs)

    def set_state(self, state):
        """
        Set which state the model is currently being used in. This can be any
        string and is used to by the cache to keep different output tensors
        for each of the states used. The default value of self.state is 'test'
        as a reference to the inference phase, but you could set it as 'train'
        (which would for instance mean that the model uses dropout layers).
        :param state: (boolean) True if the model should be in training state
                      or False for inference.
        """
        self.state = state
        print('The model state was set to: ' + state)
        return

    def fprop_layer(self, x, layer):
        """
        Expose the hidden features of a model given a layer name.
        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation of the hidden features
        """
        # Return the symbolic representation for this layer.
        return self.fprop(x)[layer]

    def fprop_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        raise NotImplementedError('`fprop_logits` not implemented')

    def fprop_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities (i.e.,
                the output values produced by the softmax layer).
        """
        raise NotImplementedError('`fprop_probs` not implemented')

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model abstraction.
        """
        raise NotImplementedError('`get_layer_names` not implemented')

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

        # The implementation (missing here because this is an abstract class)
        # should populate the dictionary with all layers returned by
        # the method self.get_layer_names()
        # assert all([layer in self.fprop_cache[(x, self.state)]
        #             for layer in self.get_layer_names()])

        raise NotImplementedError('`fprop` not implemented')


class KerasModelWrapper(Model):
    """
    An implementation of `Model` that wraps a Keras model. It
    specifically exposes the hidden features of a model by creating new models.
    The symbolic graph is reused and so there is little overhead. Splitting
    in-place operations can incur an overhead.
    """

    def __init__(self, model):
        """
        Create a wrapper for a Keras model
        :param model: A Keras model
        """
        super(KerasModelWrapper, self).__init__(model)

        # Model caching to create a new model only once for each hidden layer
        self.modelw_layer = {}
        # One model wrapper cache for `fprop`, init in the first call
        self.modelw = None

    def set_state(self, train):
        """
        Keras has its own way to keep track of the state of the graph so we'll
        throw an exception here.
        """
        raise NotImplementedError('Keras keeps track of the graph state with'
                                  'the learning phase variable. Use the'
                                  'learning phase variable rather than this'
                                  'method to change the behavior of the'
                                  'graph.')

    def _get_softmax_name(self):
        """
        Looks for a softmax layer and if found returns the output right before
        the softmax activation.

        :return: Softmax layer name
        """
        for i, layer in enumerate(self.model.layers):
            cfg = layer.get_config()
            if 'activation' in cfg and cfg['activation'] == 'softmax':
                return layer.name

        raise Exception("No softmax layers found")

    def _get_logits_name(self):
        softmax_name = self._get_softmax_name()
        softmax_layer = self.model.get_layer(softmax_name)
        node = softmax_layer.inbound_nodes[0]
        logits_name = node.inbound_layers[0].name

        return logits_name

    def fprop_logits(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the logits
        """
        logits_name = self._get_logits_name()

        return self.fprop_layer(x, logits_name)

    def fprop_probs(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the probs
        """
        name = self._get_softmax_name()

        return self.fprop_layer(x, name)

    def get_layer_names(self):
        """
        :return: Names of all the layers kept by Keras
        """
        layer_names = [x.name for x in self.model.layers]
        return layer_names

    def fprop(self, x):
        """
        Exposes all the layers of the model returned by get_layer_names.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        if (x, self.state) in self.fprop_cache.keys():
            return self.fprop_cache[(x, self.state)]
        else:
            from keras.models import Model as KerasModel
            fprop_dict = {}

            # Construct the output representation of each layer
            for layer in self.get_layer_names():
                # Get input
                new_input = self.model.get_input_at(0)
                # Find the layer to connect
                target_feat = self.model.get_layer(layer).output
                # Build a new model
                new_model = KerasModel(new_input, target_feat)
                # Add this layer's output tensor to the dictionary
                fprop_dict[layer] = new_model(x)

            # Save to cache before returning
            self.fprop_cache[(x, self.state)] = fprop_dict
            return fprop_dict
