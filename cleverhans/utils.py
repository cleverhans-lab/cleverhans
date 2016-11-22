from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import keras


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

    # When there are not enough inputs left, we reuse some to complete the batch
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

    other_classes_list = list(xrange(nb_classes))
    other_classes_list.remove(class_ind)

    return other_classes_list