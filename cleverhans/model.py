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
        pass

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
        :raise: NoSuchLayerError if `layer` is not in the model.
        """
        # Return the symbolic representation for this layer.
        output = self.fprop(x)
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested

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
        try:
            return self.get_layer(x, 'probs')
        except NoSuchLayerError:
            import tensorflow as tf
            return tf.nn.softmax(self.get_logits(x))

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model abstraction.
        """

        if hasattr(self, 'layer_names'):
            return self.layer_names

        raise NotImplementedError('`get_layer_names` not implemented.')

    def fprop(self, x):
        """
        Exposes all the layers of the model returned by get_layer_names.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        raise NotImplementedError('`fprop` not implemented.')


class CallableModelWrapper(Model):

    def __init__(self, callable_fn, output_layer):
        """
        Wrap a callable function that takes a tensor as input and returns
        a tensor as output with the given layer name.
        :param callable_fn: The callable function taking a tensor and
                            returning a given layer as output.
        :param output_layer: A string of the output layer returned by the
                             function. (Usually either "probs" or "logits".)
        """

        self.output_layer = output_layer
        self.callable_fn = callable_fn

    def get_layer_names(self):
        return [self.output_layer]

    def fprop(self, x):
        return {self.output_layer: self.callable_fn(x)}


class NoSuchLayerError(ValueError):

    """Raised when a layer that does not exist is requested."""
