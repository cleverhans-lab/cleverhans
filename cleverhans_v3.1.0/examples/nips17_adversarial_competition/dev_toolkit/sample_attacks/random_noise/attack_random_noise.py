"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from scipy.misc import imread
from scipy.misc import imsave

import tensorflow as tf


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float) / 255.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, images[i, :, :, :], format='png')


def main(_):
  """Run the sample attack"""
  eps = FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    noisy_images = x_input + eps * tf.sign(tf.random_normal(batch_shape))
    x_output = tf.clip_by_value(noisy_images, 0.0, 1.0)

    with tf.Session(FLAGS.master) as sess:
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        out_images = sess.run(x_output, feed_dict={x_input: images})
        save_images(out_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
