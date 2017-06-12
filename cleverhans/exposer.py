from abc import ABCMeta


class ModelExposer(object):
    """
    An abstract class for model exposers that expose internal representations
    (needed by "Features Adversaries" https://arxiv.org/abs/1511.05122).
    Methods of this class should return a callable object. Usually, the return
    value is just a new model object. This abstraction makes the core code
    indpendent of any specific neural network package (e.g. Keras), especially
    when a specific package does not directly expose the internal
    representations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model):
        """
        Init a exposer. If `expose_layer` is implemented, `__init__`
        should keep track of the name of the layers or `self.model` should
        provide a method for retrieving a layer.

        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's post-softmax predictions
                      (probabilities).
        """

        pass

    def expose_layer(self, layer):
        """
        Expose the hidden features of a model given a layer name.

        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation of the hidden features
        """
        error = 'Feature extraction for hidden layers not implemented'
        raise NotImplementedError(error)

    def expose_logits(self):
        """
        :return: A symbolic representation of the output logits, values before
                 softmax.
        """
        raise NotImplementedError('`expose_logits` not implemented')

    def expose_probs(self):
        """
        :return: A symbolic representation of the output probabilities, values
                 after softmax.
        """
        raise NotImplementedError('`expose_probs` not implemented')

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model exposer.
        """
        raise NotImplementedError('`get_layer_names` not implemented')

    def expose_all_layers(self):
        """
        Exposes all the layers of the model that can be exposed. This can also
        be used to expose a limited set of layers.

        :return: A dictionary with keys being layer names and values being
                 symbolic representation of the output o fcorresponding layer
        """
        raise NotImplementedError('`expose_all_layers` not implemented')


class KerasModelExposer(ModelExposer):
    """
    An implementation of `ModelExposer` that can expose a Keras
    model. It specifically exposes the hidden features of a model by creating
    new models.  The symbolic graph is reused and so there is little overhead.
    Slicing in-place operations can incur an overhead.
    """

    def __init__(self, model):
        """
        Create a exposer for a Keras model

        :param model: A Keras model
        """
        super(KerasModelExposer, self).__init__(model)

        # Initialize attributes
        self.model = model
        # Model caching to create a new model only once for each hidden layer
        self.model_slice = {}
        # One model exposer cache for `expose_all_layers`, initialized in the
        # first call
        self.model_expd = None

    def expose_layer(self, layer):
        """
        Creates a new sliced model that outputs the representation after the
        specified layer. Keras layers can be retrieved using their names.

        :param layer: The name of the hidden layer
        :return: A Keras model that has all the layers needed to compute
                 the output after `layer`.
        """
        model = self.model

        if layer in self.model_slice:
            return self.model_slice[layer]

        from keras.models import Model

        # Create an extra model that exposes the hidden layer representation
        # Get input
        new_input = model.get_input_at(0)
        # Find the layer to connect
        target_feat = model.get_layer(layer).output
        # Build a new model
        new_model = Model(new_input, target_feat)
        # Cache the new model for further expose_layer calls
        self.model_slice[layer] = new_model

        return new_model

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

    def expose_logits(self):
        """
        :return: A symbolic representation of the logits
        """
        softmax_name = self._get_softmax_layer()
        softmax_layer = self.model.get_layer(softmax_name)
        node = softmax_layer.inbound_nodes[0]
        logits_name = node.inbound_layers[0].name

        return self.expose_layer(logits_name)

    def expose_probs(self):
        """
        :return: A symbolic representation of the probs
        """
        name = self._get_softmax_layer()

        return self.expose_layer(name)

    def get_layer_names(self):
        """
        Names of all the layers kept by Keras. Outputs of `model_expd` returned
        by `expose_all_layers` are in this order.

        :return: A list of layer names.
        """
        layer_names = [x.name for x in self.model.layers]
        return layer_names

    def _create_model_expd(self):
        """
        Create the new model used by `expose_all_layers` that outputs all the
        hidden representations.

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
        model_expd = Model(new_input, outputs)

        return model_expd

    def expose_all_layers(self):
        """
        Creates a new model that has all the internal symbols as its output.
        Keras layers can be retrieved using their names.

        :return: A Keras model.
        """
        if self.model_expd is None:
            self.model_expd = self._create_model_expd()
        return self.model_expd
