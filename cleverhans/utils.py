from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt
import numpy as np
import os

if LooseVersion(keras.__version__) >= LooseVersion('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D


class _ArgsWrapper(object):
    """
    Wrapper that allows attribute access to dictionaries
    """
    def __init__(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        self.args = args

    def __getattr__(self, name):
        return self.args.get(name)


def save_model(model, dir, filename, weights_only=False):
    """
    Save Keras model
    :param model:
    :param dir:
    :param filename:
    :param weights_only:
    :return:
    """
    # If target directory does not exist, create
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Construct full path
    filepath = os.path.join(dir, filename)

    if weights_only:
        # Dump model weights
        model.save_weights(filepath)
        print("Model weights were saved to: " + filepath)
    else:
        # Dump model architecture and weights
        model.save(filepath)
        print("Model was saved to: " + filepath)


def load_model(directory, filename, weights_only=False, model=None):
    """
    Loads Keras model
    :param directory:
    :param filename:
    :return:
    """

    # If restoring model weights only, make sure model argument was given
    if weights_only:
        assert model is not None

    # Construct full path to dumped model
    filepath = os.path.join(directory, filename)

    # Check if file exists
    assert os.path.exists(filepath)

    # Return Keras model
    if weights_only:
        result = model.load_weights(filepath)
        print(result)
        return model.load_weights(filepath)
    else:
        return keras.models.load_model(filepath)


def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


def other_classes(nb_classes, class_ind):
    """
    Heper function that returns a list of class indices without one class
    :param nb_classes: number of classes in total
    :param class_ind: the class index to be omitted
    :return: list of class indices without one class
    """

    other_classes_list = list(range(nb_classes))
    other_classes_list.remove(class_ind)

    return other_classes_list


def random_targets(gt, nb_classes):
    """
    Take in the correct labels for each sample and randomly choose target
    labels from the others
    :param gt: the correct labels
    :param nb_classes: The number of classes for this model
    :return: A numpy array holding the randomly-selected target classes
    """
    if len(gt.shape) > 1:
        gt = np.argmax(gt, axis=1)

    result = np.zeros(gt.shape)

    for class_ind in xrange(nb_classes):
        in_cl = gt == class_ind
        result[in_cl] = np.random.choice(other_classes(nb_classes, class_ind))

    return np_utils.to_categorical(np.asarray(result), nb_classes)


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


def pair_visual(original, adversarial, figure=None):
    """
    This function displays two images: the original and the adversarial sample
    :param original: the original input
    :param adversarial: the input after perterbations have been applied
    :param figure: if we've already displayed images, use the same plot
    :return: the matplot figure to reuse for future samples
    """

    # Ensure our inputs are of proper shape
    assert(len(original.shape) == 2 or len(original.shape) == 3)

    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        plt.ion()
        figure = plt.figure()
        figure.canvas.set_window_title('Cleverhans: Pair Visualization')

    # Add the images to the plot
    perterbations = adversarial - original
    for index, image in enumerate((original, perterbations, adversarial)):
        figure.add_subplot(1, 3, index + 1)
        plt.axis('off')

        # If the image is 2D, then we have 1 color channel
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)

        # Give the plot some time to update
        plt.pause(0.01)

    # Draw the plot and return
    plt.show()
    return figure


def grid_visual(data):
    """
    This function displays a grid of images to show full misclassification
    :param data: grid data of the form;
        [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
    :return: if necessary, the matplot figure to reuse
    """

    # Ensure interactive mode is disabled and initialize our graph
    plt.ioff()
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')

    # Add the images to the plot
    num_cols = data.shape[0]
    num_rows = data.shape[1]
    num_channels = data.shape[4]
    current_row = 0
    for y in xrange(num_rows):
        for x in xrange(num_cols):
            figure.add_subplot(num_cols, num_rows, (x+1)+(y*num_rows))
            plt.axis('off')

            if num_channels == 1:
                plt.imshow(data[x, y, :, :, 0], cmap='gray')
            else:
                plt.imshow(data[x, y, :, :, :])

    # Draw the plot and return
    plt.show()
    return figure
