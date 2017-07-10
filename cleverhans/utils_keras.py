"""
Model construction utilities based on keras
"""
from .model import Model

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from distutils.version import LooseVersion
if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D


def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
                    space (i.e. the number output of filters in the
                    convolution)
    :param kernel_shape: (required tuple or list of 2 integers) specifies
                         the strides of the convolution along the width and
                         height.
    :param padding: (required string) can be either 'valid' (no padding around
                    input or feature map) or 'same' (pad to ensure that the
                    output feature map size is identical to the layer input)
    :param input_shape: (optional) give input shape if this is the first
                        layer of the model
    :return: the Keras layer
    """
    if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
        if input_shape is not None:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding,
                          input_shape=input_shape)
        else:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding)
    else:
        if input_shape is not None:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding,
                                 input_shape=input_shape)
        else:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding)


def cnn_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    layers = [conv_2d(nb_filters, (8, 8), (2, 2), "same",
                      input_shape=input_shape),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model


class KerasModelWrapper(Model):
    """
    An implementation of `Model` that wraps a Keras model. It
    specifically exposes the hidden features of a model by creating new models.
    The symbolic graph is reused and so there is little overhead. Splitting
    in-place operations can incur an overhead.
    """

    def __init__(self, model=None):
        """
        Create a wrapper for a Keras model
        :param model: A Keras model
        """
        super(KerasModelWrapper, self).__init__()

        if model is None:
            raise ValueError('model argument must be supplied.')

        self.model = model
        self.keras_model = None

    def _get_softmax_name(self):
        """
        Looks for the name of the softmax layer.
        :return: Softmax layer name
        """
        for i, layer in enumerate(self.model.layers):
            cfg = layer.get_config()
            if 'activation' in cfg and cfg['activation'] == 'softmax':
                return layer.name

        raise Exception("No softmax layers found")

    def _get_logits_name(self):
        """
        Looks for the name of the layer producing the logits.
        :return: name of layer producing the logits
        """
        softmax_name = self._get_softmax_name()
        softmax_layer = self.model.get_layer(softmax_name)
        node = softmax_layer.inbound_nodes[0]
        logits_name = node.inbound_layers[0].name

        return logits_name

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the logits
        """
        logits_name = self._get_logits_name()

        return self.get_layer(x, logits_name)

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input.
        :return: A symbolic representation of the probs
        """
        name = self._get_softmax_name()

        return self.get_layer(x, name)

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
        from keras.models import Model as KerasModel

        if self.keras_model is None:
            # Get the input layer
            new_input = self.model.get_input_at(0)

            # Make a new model that returns each of the layers as output
            out_layers = [x_layer.output for x_layer in self.model.layers]
            self.keras_model = KerasModel(new_input, out_layers)

        # and get the outputs for that model on the input x
        outputs = self.keras_model(x)

        # Keras only returns a list for outputs of length >= 1, if the model
        # is only one layer, wrap a list
        if len(self.model.layers) == 1:
            outputs = [outputs]

        # compute the dict to return
        fprop_dict = dict(zip(self.get_layer_names(), outputs))

        return fprop_dict
