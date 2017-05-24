from abc import ABCMeta
from keras.models import Model


class FeatureExposer:
    """
    Abstract base class for Neural Network models that exposes
    hidden activations.
    """
    __meta_class__ = ABCMeta

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


class KerasFeatureExposer(FeatureExposer):
    """
    This class exposes the hidden layer features for a Keras based model
    required by the feature-adversary attack.
    """

    def __init__(self, model, layer):
        """
        Create a model abstraction from a Keras model
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
        return self.model_feat(x)
