from abc import ABCMeta


class Model(object):
    """
    An abstract interface for model wrappers that exposes model symbols
    needed for making an attack. This abstraction removes the dependency on
    any specific neural network package (e.g. Keras) from the core
    code of CleverHans. It can also simplify exposing the hidden features of a
    model when a specific package does not directly expose them (needed by
    "Features Adversaries" https://arxiv.org/abs/1511.05122).
    """
    __metaclass__ = ABCMeta

    def __init__(self, model):
        """
        Init a wrapper. If `get_layer` is implemented, `__init__`
        should keep track of the name of the layers or `self.model` should
        provide a method for retrieving a layer.

        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's post-softmax predictions
                      (probabilities).
        """

        pass

    def get_layer(self, x, layer):
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
        raise NotImplementedError('`get_logits` not implemented')

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities, values
                 after softmax.
        """
        raise NotImplementedError('`get_probs` not implemented')

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model wrapper.
        """
        raise NotImplementedError('`get_layer_names` not implemented')

    def fprop(self, x):
        """
        Exposes all the layers of the model that can be exposed. This can also
        be used to expose a limited set of layers.

        :param x: A symbolic representation of the network input
        :return: A dictionary with keys being layer names and values being
                 symbolic representation of the output o fcorresponding layer
        """
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

        # Initialize attributes
        self.model = model
        # Model caching to create a new model only once for each hidden layer
        self.modelw_layer = {}
        # One model wrapper cache for `fprop`
        self.modelw = self._create_modelw()

    def get_layer(self, x, layer):
        """
        Creates a new model with the `x` as the input and the output after the
        specified layer. Keras layers can be retrieved using their names.

        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer
        :return: A symbolic representation of the hidden features
        """
        model = self.model

        if layer in self.modelw_layer:
            return self.modelw_layer[layer](x)

        from keras.models import Model

        # Create an extra model that exposes the hidden layer representation
        # Get input
        new_input = model.get_input_at(0)
        # Find the layer to connect
        target_feat = model.get_layer(layer).output
        # Build a new model
        new_model = Model(new_input, target_feat)
        # Cache the new model for further get_layer calls
        self.modelw_layer[layer] = new_model

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
                return layer.name

        raise Exception("No softmax layers found")

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the logits
        """
        softmax_name = self._get_softmax_layer()
        softmax_layer = self.model.get_layer(softmax_name)
        node = softmax_layer.inbound_nodes[0]
        logits_name = node.inbound_layers[0].name

        return self.get_layer(x, logits_name)

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the probs
        """
        name = self._get_softmax_layer()

        return self.get_layer(x, name)

    def get_layer_names(self):
        """
        :return: Names of all the layers kept by Keras
        """
        layer_names = [x.name for x in self.model.layers]
        return layer_names

    def _create_modelw(self):
        """
        Create the new model used by fprop that outputs all the hidden outputs

        :return: A new Keras model
        """
        model = self.model

        from keras.models import Model

        # Get input
        new_input = model.get_input_at(0)
        # Collect the output symbols for all the layers
        layer_names = self.get_layer_names()
        outputs = [model.get_layer(name).output for name in layer_names]
        # Build a new model
        modelw = Model(new_input, outputs)

        return modelw

    def fprop(self, x):
        """
        Creates a new model with the `x` as the input and the output after the
        specified layer. Keras layers can be retrieved using their names.

        :param x: A symbolic representation of the network input
        :return: A dictionary with keys being layer names and values being
                 symbolic representation of the output o fcorresponding layer
        """
        layer_names = self.get_layer_names()
        outputs = self.modelw(x)
        out_dict = dict(zip(layer_names, outputs))
        return out_dict
