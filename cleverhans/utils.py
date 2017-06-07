from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import warnings
from six.moves import xrange


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


class AccuracyReport(object):

    """
    An object summarizing the accuracy results for experiments involving
    training on clean examples or adversarial examples, then evaluating
    on clean or adversarial examples.
    """

    def __init__(self):
        self.clean_train_clean_eval = 0.
        self.clean_train_adv_eval = 0.
        self.adv_train_clean_eval = 0.
        self.adv_train_adv_eval = 0.


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


def pair_visual(original, adversarial, figure=None):
    """
    This function displays two images: the original and the adversarial sample
    :param original: the original input
    :param adversarial: the input after perterbations have been applied
    :param figure: if we've already displayed images, use the same plot
    :return: the matplot figure to reuse for future samples
    """
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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
            figure.add_subplot(num_cols, num_rows, (x + 1) + (y * num_rows))
            plt.axis('off')

            if num_channels == 1:
                plt.imshow(data[x, y, :, :, 0], cmap='gray')
            else:
                plt.imshow(data[x, y, :, :, :])

    # Draw the plot and return
    plt.show()
    return figure


def conv_2d(*args, **kwargs):
    from cleverhans.utils_keras import conv_2d
    warnings.warn("utils.conv_2d is deprecated and may be removed on or after"
                  " 2018-01-05. Switch to utils_keras.conv_2d.")
    return conv_2d(*args, **kwargs)


def cnn_model(*args, **kwargs):
    from cleverhans.utils_keras import cnn_model
    warnings.warn("utils.cnn_model is deprecated and may be removed on or"
                  " after 2018-01-05. Switch to utils_keras.cnn_model.")
    return cnn_model(*args, **kwargs)
