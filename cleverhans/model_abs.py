from abc import ABCMeta
from keras.models import Model


class ModelAbstraction:
    """
    An abstract interface for a model that exposes the required functionalities
    """
    __metaclass__ = ABCMeta

    def __init__(self, model):
        """
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        """
        raise NotImplementedError('Model abstraction not implementated.')

    def get_hidden(self, layer):
        """
        :return: A symbolic representation of the hidden features
        :param layer: The name of the hidden layer to return features at.
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


class KerasModelWrapper(ModelAbstraction):
    """
    An implementation of FeatureExposer that wrapps a Keras model.
    """

    def __init__(self, model):
        """
        Create a wrapper from a Keras model
        :param model: A Keras model
        """
        super(KerasModelWrapper, self).__init__(model)

        # Initialize attributes
        self.model = model

    def get_hidden(self, layer):
        """
        :param layer: The name of the hidden layer
        :return: A symbolic representation of the hidden features
        """
        model = self.model

        # Create an extra model that exposes the hidden layer representation
        # Get input
        new_input = model.get_input_at(0)
        # Find the layer to connect
        target_feat = model.get_layer(layer).output
        # Build a new model
        h = Model(new_input, target_feat)

        return h
