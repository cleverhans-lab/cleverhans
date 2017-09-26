# started from CIFAR10 in RevNets
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

import numpy as np
import tensorflow as tf  # NOQA
import scipy.io as sio

# Global constants describing the SVHN data set.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CLASSES = 10
NUM_CHANNEL = 3
NUM_TRAIN_IMG = 73257+531131
NUM_TEST_IMG = 26032


def read_SVHN(data_folder):
    """ Reads and parses examples from SVHN data files """

    train_img = []
    train_label = []
    test_img = []
    test_label = []

    train_file_list = [
        'train_32x32.mat', 'extra_32x32.mat'
    ]
    test_file_list = ["test_32x32.mat"]

    for i in xrange(len(train_file_list)):
        tmp_dict = sio.loadmat(os.path.join(data_folder, train_file_list[i]))
        train_img.append(tmp_dict["X"])
        train_label.append(tmp_dict["y"])

    tmp_dict = sio.loadmat(
        os.path.join(data_folder, test_file_list[0]))
    test_img.append(tmp_dict["X"])
    test_label.append(tmp_dict["y"])

    train_img = np.concatenate(train_img, axis=-1)
    train_label = np.concatenate(train_label).flatten()
    test_img = np.concatenate(test_img, axis=-1)
    test_label = np.concatenate(test_label).flatten()

    # change format from [H, W, C, B] to [B, H, W, C] for feeding to Tensorflow
    train_img = np.transpose(train_img, [3, 0, 1, 2])
    test_img = np.transpose(test_img, [3, 0, 1, 2])

    mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

    train_img = train_img - mean_img
    test_img = test_img - mean_img
    train_y = train_label - 1  # 0-based label
    test_y = test_label - 1    # 0-based label

    train_label = np.eye(10)[train_y]
    test_label = np.eye(10)[test_y]

    return train_img, train_label, test_img, test_label


def svhn_tf_preprocess(inp, random_crop=True):
    image_size = 32
    # inp = tf.placeholder(tf.float32, [image_size, image_size, 3])
    image = inp
    # image = tf.cast(inp, tf.float32)
    if random_crop:
        print("Apply random cropping")
        image = tf.image.resize_image_with_crop_or_pad(inp, image_size + 4,
                                                       image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, 3])
    # if random_flip:
    #     print("Apply random flipping")
    #     image = tf.image.random_flip_left_right(image)
    # Brightness/saturation/constrast provides small gains .2%~.5% on svhn.
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # if whiten:
    #     print("Apply whitening")
    #     image = tf.image.per_image_whitening(image)
    return inp, image
