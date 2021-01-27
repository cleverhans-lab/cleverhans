"""
https://github.com/renmengye/revnet-public/blob/master/resnet/data/cifar_input.py

MIT License

Copyright (c) 2017 Mengye Ren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# pylint: disable=missing-docstring
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import cPickle as pkl

import numpy as np
from six.moves import range
import tensorflow as tf

# Global constants describing the CIFAR-10 data set.
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CLASSES = 10
NUM_CHANNEL = 3
NUM_TRAIN_IMG = 50000
NUM_TEST_IMG = 10000


def unpickle(file):
  fo = open(file, 'rb')
  dict = pkl.load(fo)
  fo.close()
  return dict


def read_CIFAR10(data_folder):
  """ Reads and parses examples from CIFAR10 data files """

  train_img = []
  train_label = []
  test_img = []
  test_label = []

  train_file_list = [
      "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
      "data_batch_5"
  ]
  test_file_list = ["test_batch"]

  if "CIFAR10_PATH" in os.environ:
    cifar10_path = os.environ["CIFAR10_PATH"]
  else:
    cifar10_path = 'cifar-10-batches-py'

  for i in range(len(train_file_list)):
    tmp_dict = unpickle(os.path.join(data_folder, cifar10_path,
                                     train_file_list[i]))

    train_img.append(tmp_dict["data"])
    train_label.append(tmp_dict["labels"])

  tmp_dict = unpickle(
      os.path.join(data_folder, cifar10_path, test_file_list[0]))
  test_img.append(tmp_dict["data"])
  test_label.append(tmp_dict["labels"])

  train_img = np.concatenate(train_img)
  train_label = np.concatenate(train_label)
  test_img = np.concatenate(test_img)
  test_label = np.concatenate(test_label)

  train_img = np.reshape(
      train_img, [NUM_TRAIN_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])
  test_img = np.reshape(
      test_img, [NUM_TEST_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])

  # change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
  train_img = np.transpose(train_img, [0, 2, 3, 1])
  test_img = np.transpose(test_img, [0, 2, 3, 1])

  mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

  CIFAR10_data = {}
  CIFAR10_data["train_img"] = train_img - mean_img
  CIFAR10_data["test_img"] = test_img - mean_img
  CIFAR10_data["train_label"] = train_label
  CIFAR10_data["test_label"] = test_label

  train_img = train_img - mean_img
  test_img = test_img - mean_img
  train_label = train_label
  test_label = test_label

  train_label = np.eye(10)[train_label]
  test_label = np.eye(10)[test_label]

  return train_img, train_label, test_img, test_label


def read_CIFAR100(data_folder):
  """ Reads and parses examples from CIFAR100 python data files """

  train_img = []
  train_label = []
  test_img = []
  test_label = []

  train_file_list = ["cifar-100-python/train"]
  test_file_list = ["cifar-100-python/test"]

  tmp_dict = unpickle(os.path.join(data_folder, train_file_list[0]))
  train_img.append(tmp_dict["data"])
  train_label.append(tmp_dict["fine_labels"])

  tmp_dict = unpickle(os.path.join(data_folder, test_file_list[0]))
  test_img.append(tmp_dict["data"])
  test_label.append(tmp_dict["fine_labels"])

  train_img = np.concatenate(train_img)
  train_label = np.concatenate(train_label)
  test_img = np.concatenate(test_img)
  test_label = np.concatenate(test_label)

  train_img = np.reshape(
      train_img, [NUM_TRAIN_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])
  test_img = np.reshape(
      test_img, [NUM_TEST_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])

  # change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
  train_img = np.transpose(train_img, [0, 2, 3, 1])
  test_img = np.transpose(test_img, [0, 2, 3, 1])
  mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

  CIFAR100_data = {}
  CIFAR100_data["train_img"] = train_img - mean_img
  CIFAR100_data["test_img"] = test_img - mean_img
  CIFAR100_data["train_label"] = train_label
  CIFAR100_data["test_label"] = test_label

  return CIFAR100_data


def cifar_tf_preprocess(inp, random_crop=True, random_flip=True, whiten=True,
                        br_sat_con=False):
  image_size = 32
  image = inp
  if random_crop:
    image = tf.image.resize_image_with_crop_or_pad(inp, image_size + 4,
                                                   image_size + 4)
    image = tf.random_crop(image, [image_size, image_size, 3])
  if random_flip:
    image = tf.image.random_flip_left_right(image)
  # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
  if br_sat_con:
    image = tf.image.random_brightness(image, max_delta=63. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
  if whiten:
    image = tf.image.per_image_standardization(image)
  return image
