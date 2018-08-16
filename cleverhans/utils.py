from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from collections import OrderedDict
from six.moves import xrange
import warnings
import logging

known_number_types = (int, float, np.float16, np.float32, np.float64,
                      np.int8, np.int16, np.int32, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64)


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

        # Training data accuracy results to be used by tutorials
        self.train_clean_train_clean_eval = 0.
        self.train_clean_train_adv_eval = 0.
        self.train_adv_train_clean_eval = 0.
        self.train_adv_train_adv_eval = 0.


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
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: list of class indices excluding the class indexed by class_ind
    """
    if class_ind < 0 or class_ind >= nb_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_classes_list = list(range(nb_classes))
    other_classes_list.remove(class_ind)

    return other_classes_list


def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    This is adapted from the Keras function with the same name.
    :param y: class vector to be converted into a matrix
              (integers from 0 to num_classes).
    :param num_classes: num_classes: total number of classes.
    :return: A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
        warnings.warn("FutureWarning: the default value of the second"
                      "argument in function \"to_categorical\" is deprecated."
                      "On 2018-9-19, the second argument"
                      "will become mandatory.")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def random_targets(gt, nb_classes):
    """
    Take in an array of correct labels and randomly select a different label
    for each label in the array. This is typically used to randomly select a
    target class in targeted adversarial examples attacks (i.e., when the
    search algorithm takes in both a source class and target class to compute
    the adversarial example).
    :param gt: the ground truth (correct) labels. They can be provided as a
               1D vector or 2D array of one-hot encoded labels.
    :param nb_classes: The number of classes for this task. The random class
                       will be chosen between 0 and nb_classes such that it
                       is different from the correct class.
    :return: A numpy array holding the randomly-selected target classes
             encoded as one-hot labels.
    """
    # If the ground truth labels are encoded as one-hot, convert to labels.
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)

    # This vector will hold the randomly selected labels.
    result = np.zeros(gt.shape, dtype=np.int32)

    for class_ind in xrange(nb_classes):
        # Compute all indices in that class.
        in_cl = gt == class_ind
        size = np.sum(in_cl)

        # Compute the set of potential targets for this class.
        potential_targets = other_classes(nb_classes, class_ind)

        # Draw with replacement random targets among the potential targets.
        result[in_cl] = np.random.choice(potential_targets, size=size)

    # Encode vector of random labels as one-hot labels.
    result = to_categorical(result, nb_classes)
    result = result.astype(np.int32)

    return result


def pair_visual(original, adversarial, figure=None):
    """
    This function displays two images: the original and the adversarial sample
    :param original: the original input
    :param adversarial: the input after perterbations have been applied
    :param figure: if we've already displayed images, use the same plot
    :return: the matplot figure to reuse for future samples
    """
    import matplotlib.pyplot as plt

    # Squeeze the image to remove single-dimensional entries from array shape
    original = np.squeeze(original)
    adversarial = np.squeeze(adversarial)

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
            figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
            plt.axis('off')

            if num_channels == 1:
                plt.imshow(data[x, y, :, :, 0], cmap='gray')
            else:
                plt.imshow(data[x, y, :, :, :])

    # Draw the plot and return
    plt.show()
    return figure


def get_logits_over_interval(sess, model, x_data, fgsm_params,
                             min_epsilon=-10., max_epsilon=10.,
                             num_points=21):
    """Get logits when the input is perturbed in an interval in adv direction.

    Args:
        sess: Tf session
        model: Model for which we wish to get logits.
        x_data: Numpy array corresponding to single data.
                point of shape [height, width, channels].
        fgsm_params: Parameters for generating adversarial examples.
        min_epsilon: Minimum value of epsilon over the interval.
        max_epsilon: Maximum value of epsilon over the interval.
        num_points: Number of points used to interpolate.

    Returns:
        Numpy array containing logits.

    Raises:
        ValueError if min_epsilon is larger than max_epsilon.
    """
    # Get the height, width and number of channels
    height = x_data.shape[0]
    width = x_data.shape[1]
    channels = x_data.shape[2]
    size = height * width * channels

    x_data = np.expand_dims(x_data, axis=0)
    import tensorflow as tf
    from cleverhans.attacks import FastGradientMethod

    # Define the data placeholder
    x = tf.placeholder(dtype=tf.float32,
                       shape=[1, height,
                              width,
                              channels],
                       name='x')
    # Define adv_x
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)

    if min_epsilon > max_epsilon:
        raise ValueError('Minimum epsilon is less than maximum epsilon')

    eta = tf.nn.l2_normalize(adv_x - x, dim=0)
    epsilon = tf.reshape(tf.lin_space(float(min_epsilon),
                                      float(max_epsilon),
                                      num_points),
                         (num_points, 1, 1, 1))
    lin_batch = x + epsilon * eta
    logits = model.get_logits(lin_batch)
    with sess.as_default():
        log_prob_adv_array = sess.run(logits,
                                      feed_dict={x: x_data})
    return log_prob_adv_array


def linear_extrapolation_plot(log_prob_adv_array, y, file_name,
                              min_epsilon=-10, max_epsilon=10,
                              num_points=21):
    """Generate linear extrapolation plot.

    Args:
        log_prob_adv_array: Numpy array containing log probabilities
        y: Tf placeholder for the labels
        file_name: Plot filename
        min_epsilon: Minimum value of epsilon over the interval
        max_epsilon: Maximum value of epsilon over the interval
        num_points: Number of points used to interpolate
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Linear Extrapolation Plot')

    correct_idx = np.argmax(y, axis=0)
    fig = plt.figure()
    plt.xlabel('Epsilon')
    plt.ylabel('Logits')
    x_axis = np.linspace(min_epsilon, max_epsilon, num_points)
    plt.xlim(min_epsilon - 1, max_epsilon + 1)
    for i in xrange(y.shape[0]):
        if i == correct_idx:
            ls = '-'
            linewidth = 5
        else:
            ls = '--'
            linewidth = 2
        plt.plot(
            x_axis,
            log_prob_adv_array[:, i],
            ls=ls,
            linewidth=linewidth,
            label='{}'.format(i))
    plt.legend(loc='best', fontsize=14)
    plt.show()
    fig.savefig(file_name)
    plt.clf()
    return figure


def set_log_level(level, name="cleverhans"):
    """
    Sets the threshold for the cleverhans logger to level
    :param level: the logger threshold. You can find values here:
                  https://docs.python.org/2/library/logging.html#levels
    :param name: the name used for the cleverhans logger
    """
    logging.getLogger(name).setLevel(level)


def get_log_level(name="cleverhans"):
    """
    Gets the current threshold for the cleverhans logger
    :param name: the name used for the cleverhans logger
    """
    return logging.getLogger(name).getEffectiveLevel()


class TemporaryLogLevel(object):
    """
    A ContextManager that changes a log level temporarily.

    Note that the log level will be set back to its original value when
    the context manager exits, even if the log level has been changed
    again in the meantime.
    """

    def __init__(self, level, name):
        self.name = name
        self.level = level

    def __enter__(self):
        self.old_level = get_log_level(self.name)
        set_log_level(self.level, self.name)

    def __exit__(self, type, value, traceback):
        set_log_level(self.old_level, self.name)
        return True


def create_logger(name):
    """
    Create a logger object with the given name.

    If this is the first time that we call this method, then initialize the
    formatter.
    """
    base = logging.getLogger("cleverhans")
    if len(base.handlers) == 0:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] ' +
                                      '%(message)s')
        ch.setFormatter(formatter)
        base.addHandler(ch)

    return base


def deterministic_dict(normal_dict):
    """
    Returns a version of `normal_dict` whose iteration order is always the same
    """
    out = OrderedDict()
    for key in sorted(normal_dict.keys()):
        out[key] = normal_dict[key]
    return out
