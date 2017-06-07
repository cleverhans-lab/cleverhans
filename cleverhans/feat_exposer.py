from abc import ABCMeta
from keras.models import Model


class FeatureExposer:
    """
    An abstract interface for a model that exposes hidden activations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, layer):
        """
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        :param layer: The name of the hidden layer to compute features at.
        """
        raise NotImplementedError('Model abstraction not implementated.')

    def compute_feat(self, x):
        """
        :param x: The model's symbolic inputs
        :return: A symbolic representation of the hidden features
        """
        error = 'Feature extraction for hidden layers not implemented'
        raise NotImplementedError(error)

    def get_logits(self):
        """
        :return: A symbolic representation of the output logits, values before
        softmax
        """
        error = 'get_logits not implemented'
        raise NotImplementedError(error)

    def get_probs(self):
        """
        :return: A symbolic representation of the output logits, values before
        softmax
        """
        error = 'get_probs not implemented'
        raise NotImplementedError(error)


class KerasFeatureExposer(FeatureExposer):
    """
    An implementation of FeatureExposer that wrapps a Keras model.
    """

    def __init__(self, model, layer):
        """
        Create a feature exposer from a Keras model
        :param model: A Keras model
        :param layer: The name of the hidden layer
        """
        super(KerasFeatureExposer, self).__init__(model)

        # Initialize attributes
        self.model = model
        self.layer = layer

        # Create an extra model that exposes the hidden layer representation
        # Get input
        new_input = model.get_input_at(0)
        # Find the layer to connect
        target_feat = model.get_layer(layer).output
        # Build a new model
        self.model_feat = Model(new_input, target_feat)

    def compute_feat(self, x):
        """
        :param x: The model's symbolic inputs
        :return: A symbolic representation of the hidden features
        """
        return self.model_feat(x)
