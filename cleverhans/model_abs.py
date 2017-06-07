from abc import ABCMeta


class Model(object):
    """
    An abstract interface for model wrappers that exposes model symbols
    needed for making an attack. This abstraction removes the dependency on
    any specific neural network package (e.g. Keras) from the core
    code of CleverHans. It can also simplify exposing the hidden features of a
    model when a specific package does not directly expose them (needed by
    "features adversaries").
    """
    __metaclass__ = ABCMeta

    def __init__(self, model):
        """
        Init a wrapper. If `fprop` is implemented, `__init__`
        should keep track of the name of the layers or `self.model` should
        provide a method for retrieving a layer.

        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        """

        pass

    def fprop(self, x, layer=None):
        """
        Expose the hidden features of a model given a layer name.

        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation of the hidden features
        """
        error = 'Feature extraction for hidden layers not implemented'
        raise NotImplementedError(error)

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits, values before
                 softmax.
        """
        error = '`get_logits` not implemented'
        raise NotImplementedError(error)

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities, values
                 after softmax.
        """
        error = '`get_probs` not implemented'
        raise NotImplementedError(error)


class KerasModelWrapper(Model):
    """
    An implementation of ModelAbstraction that wraps a Keras model. It
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

        # Initialize attributes
        self.model = model
        self.model_dict = {None: model}

    def fprop(self, x, layer=None):
        """
        Creates a new model with the `x` as the input and the output after the
        specified layer. Keras layers can be retrieved using their names.

        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer
        :return: A symbolic representation of the hidden features
        """
        model = self.model

        if layer in self.model_dict:
            return self.model_dict(x)

        from keras.models import Model

        # Create an extra model that exposes the hidden layer representation
        # Get input
        new_input = model.get_input_at(0)
        # Find the layer to connect
        target_feat = model.get_layer(layer).output
        # Build a new model
        new_model = Model(new_input, target_feat)
        # Cache the new model for further fprop calls
        self.model_dict[layer] = new_model

        return new_model(x)

    def _get_softmax_layer(self):
        """
        Looks for a softmax layer and if found returns the output right before
        the softmax activation.

        :return: Softmax layer name
        """
        for i, layer in enumerate(self.model.layers):
            cfg = layer.get_config()
            if 'activation' in cfg and cfg['activation'] == 'softmax':
                return cfg.name

        raise Exception("No softmax layers found")

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the logits
        """
        softmax_name = self._get_softmax_layer()
        softmax_layer = self.model.get_layer(softmax_name)
        node = softmax_layer.inbound_nodes[0]
        logits_name = node.inbound_layers[0]

        return self.fprop(x, logits_name)

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the probs
        """
        name = self._get_softmax_layer()

        return self.fprop(x, name)
