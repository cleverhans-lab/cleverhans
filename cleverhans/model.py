from abc import ABCMeta
import tensorflow as tf


class Model(object):
    """
    An abstract interface for model wrappers that exposes model symbols
    needed for making an attack. This abstraction removes the dependency on
    any specific neural network package (e.g. Keras) from the core
    code of CleverHans. It can also simplify exposing the hidden features of a
    model when a specific package does not directly expose them.
    """
    __metaclass__ = ABCMeta
    O_LOGITS, O_PROBS, O_FEATURES = 'logits probs features'.split()

    def __init__(self, scope=None, nb_classes=10, hparams=None):
        """
        Constructor.
        :param scope: str, the name of model.
        :param nb_classes: integer, the number of classes.
        :param hparams: dict, hyper-parameters for the model.
        """
        self.scope = scope or self.__class__.__name__
        self.nb_classes = nb_classes
        self.hparams = hparams or {}

    def __call__(self, *args, **kwargs):
        """
        For compatibility with functions used as model definitions (taking
        an input tensor and returning the tensor giving the output
        of the model on that input).
        """
        return self.get_probs(*args, **kwargs)

    def get_logits(self, x, **kwargs):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the output logits
        (i.e., the values fed as inputs to the softmax layer).
        """
        return self.fprop(x, **kwargs)[self.O_LOGITS]

    def get_probs(self, x, **kwargs):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the output
        probabilities (i.e., the output values produced by the softmax layer).
        """
        d = self.fprop(x, **kwargs)
        if self.O_PROBS in d:
            return d[self.O_PROBS]
        elif self.O_LOGITS in d:
            return tf.nn.softmax(logits=d[self.O_LOGITS])
        else:
            raise ValueError('Cannot find probs or logits.')

    def fprop(self, x, **kwargs):
        """
        Forward propagation to compute the model outputs.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        raise NotImplementedError('`fprop` not implemented.')

    def get_params(self):
        """
        Provides access to the model's parameters.
        :return: A list of all Variables defining the model parameters.
        """
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       self.scope)
        return scope_vars

    def get_layer_names(self):
        """Return the list of exposed layers for this model."""
        raise NotImplementedError

    def get_layer(self, x, layer, **kwargs):
        """Return a layer output.
        :param x: tensor, the input to the network.
        :param layer: str, the name of the layer to compute.
        :param **kwargs: dict, extra optional params to pass to self.fprop.
        :return: the content of layer `layer`
        """
        return self.fprop(x, **kwargs)[layer]


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

    def fprop(self, x, **kwargs):
        return {self.output_layer: self.callable_fn(x, **kwargs)}


class NoSuchLayerError(ValueError):

    """Raised when a layer that does not exist is requested."""
