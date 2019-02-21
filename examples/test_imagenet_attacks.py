"""Test attack success against ImageNet models for a few images.

Many of the tests require using flags to specify a pre-trained ImageNet model,
as well as image data. The easiest way to provide these is using the data from
cleverhans/examples/nips17_adversarial_competition, and then the default flag
values will just work.

Setup: see SETUP_INSTRUCTIONS
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import os
import unittest

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib import slim
# The following line is affected by a pylint bug when using python3 and tf 1.12
from tensorflow.contrib.slim.nets import inception # pylint: disable=no-name-in-module
from PIL import Image
from cleverhans.attacks import SPSA
from cleverhans.devtools.checks import CleverHansTest
from cleverhans.model import Model
from cleverhans.utils import CLEVERHANS_ROOT

SETUP_INSTRUCTIONS = """
$ ./examples/nips17_adversarial_competition/dev_toolkit/download_data.sh
"""

DEFAULT_INCEPTION_PATH = os.path.join(
    CLEVERHANS_ROOT,
    ('examples/nips17_adversarial_competition/dev_toolkit/sample_attacks/fgsm/'
     'inception_v3.ckpt'))

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path',
    DEFAULT_INCEPTION_PATH, 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_image_dir',
    os.path.join(CLEVERHANS_ROOT,
                 'examples/nips17_adversarial_competition/dataset/images'),
    'Path to image directory.')

tf.flags.DEFINE_string(
    'metadata_file_path',
    os.path.join(
        CLEVERHANS_ROOT,
        'examples/nips17_adversarial_competition/dataset/dev_dataset.csv'),
    'Path to metadata file.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, metadata_file_path, batch_shape):
  """Retrieve numpy arrays of images and labels, read from a directory."""
  num_images = batch_shape[0]
  with open(metadata_file_path) as input_file:
    reader = csv.reader(input_file)
    header_row = next(reader)
    rows = list(reader)

  row_idx_image_id = header_row.index('ImageId')
  row_idx_true_label = header_row.index('TrueLabel')
  images = np.zeros(batch_shape)
  labels = np.zeros(num_images, dtype=np.int32)
  for idx in xrange(num_images):
    row = rows[idx]
    filepath = os.path.join(input_dir, row[row_idx_image_id] + '.png')

    with tf.gfile.Open(filepath, 'rb') as f:
      image = np.array(
          Image.open(f).convert('RGB')).astype(np.float) / 255.0
    images[idx, :, :, :] = image
    labels[idx] = int(row[row_idx_true_label])
  return images, labels


class InceptionModel(Model):
  """Model class for CleverHans library."""

  def __init__(self, nb_classes):
    super(InceptionModel, self).__init__(nb_classes=nb_classes,
                                         needs_dummy_fprop=True)
    self.built = False

  def __call__(self, x_input, return_logits=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      # Inception preprocessing uses [-1, 1]-scaled input.
      x_input = x_input * 2.0 - 1.0
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.nb_classes, is_training=False,
          reuse=reuse)
    self.built = True
    self.logits = end_points['Logits']
    # Strip off the extra reshape op at the output
    self.probs = end_points['Predictions'].op.inputs[0]
    if return_logits:
      return self.logits
    else:
      return self.probs

  def get_logits(self, x_input):
    return self(x_input, return_logits=True)

  def get_probs(self, x_input):
    return self(x_input)


def _top_1_accuracy(logits, labels):
  return tf.reduce_mean(
      tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))


class TestInception(CleverHansTest):
  def test_clean_accuracy(self):
    """Check model is accurate on unperturbed images."""
    input_dir = FLAGS.input_image_dir
    metadata_file_path = FLAGS.metadata_file_path
    num_images = 16
    batch_shape = (num_images, 299, 299, 3)
    images, labels = load_images(
        input_dir, metadata_file_path, batch_shape)
    nb_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
      # Prepare graph
      x_input = tf.placeholder(tf.float32, shape=batch_shape)
      y_label = tf.placeholder(tf.int32, shape=(num_images,))
      model = InceptionModel(nb_classes)
      logits = model.get_logits(x_input)
      acc = _top_1_accuracy(logits, y_label)

      # Run computation
      saver = tf.train.Saver(slim.get_model_variables())

      session_creator = tf.train.ChiefSessionCreator(
          scaffold=tf.train.Scaffold(saver=saver),
          checkpoint_filename_with_path=FLAGS.checkpoint_path,
          master=FLAGS.master)

      with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        acc_val = sess.run(acc, feed_dict={x_input: images, y_label: labels})
        tf.logging.info('Accuracy: %s', acc_val)
        assert acc_val > 0.8


class TestSPSA(CleverHansTest):
  def test_attack_bounds(self):
    """Check SPSA respects perturbation limits."""
    epsilon = 4. / 255
    input_dir = FLAGS.input_image_dir
    metadata_file_path = FLAGS.metadata_file_path
    num_images = 8
    batch_shape = (num_images, 299, 299, 3)
    images, labels = load_images(
        input_dir, metadata_file_path, batch_shape)
    nb_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
      # Prepare graph
      x_input = tf.placeholder(tf.float32, shape=(1,) + batch_shape[1:])
      y_label = tf.placeholder(tf.int32, shape=(1,))
      model = InceptionModel(nb_classes)

      attack = SPSA(model)
      x_adv = attack.generate(
          x_input, y=y_label, epsilon=epsilon, num_steps=10,
          early_stop_loss_threshold=-1., spsa_samples=32, spsa_iters=1,
          is_debug=True)

      # Run computation
      saver = tf.train.Saver(slim.get_model_variables())
      session_creator = tf.train.ChiefSessionCreator(
          scaffold=tf.train.Scaffold(saver=saver),
          checkpoint_filename_with_path=FLAGS.checkpoint_path,
          master=FLAGS.master)

      with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        for i in xrange(num_images):
          x_expanded = np.expand_dims(images[i], axis=0)
          y_expanded = np.expand_dims(labels[i], axis=0)

          adv_image = sess.run(x_adv, feed_dict={x_input: x_expanded,
                                                 y_label: y_expanded})
          diff = adv_image - images[i]
          assert np.max(np.abs(diff)) < epsilon + 1e-4
          assert np.max(adv_image < 1. + 1e-4)
          assert np.min(adv_image > -1e-4)

  def test_attack_success(self):
    """Check SPSA creates misclassified images."""
    epsilon = 4. / 255
    input_dir = FLAGS.input_image_dir
    metadata_file_path = FLAGS.metadata_file_path
    num_images = 8
    batch_shape = (num_images, 299, 299, 3)
    images, labels = load_images(
        input_dir, metadata_file_path, batch_shape)
    nb_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
      # Prepare graph
      x_input = tf.placeholder(tf.float32, shape=(1,) + batch_shape[1:])
      y_label = tf.placeholder(tf.int32, shape=(1,))
      model = InceptionModel(nb_classes)

      attack = SPSA(model)
      x_adv = attack.generate(
          x_input, y=y_label, epsilon=epsilon, num_steps=30,
          early_stop_loss_threshold=-1., spsa_samples=32, spsa_iters=16,
          is_debug=True)

      logits = model.get_logits(x_adv)
      acc = _top_1_accuracy(logits, y_label)

      # Run computation
      saver = tf.train.Saver(slim.get_model_variables())
      session_creator = tf.train.ChiefSessionCreator(
          scaffold=tf.train.Scaffold(saver=saver),
          checkpoint_filename_with_path=FLAGS.checkpoint_path,
          master=FLAGS.master)

      num_correct = 0.
      with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        for i in xrange(num_images):
          feed_dict_i = {x_input: np.expand_dims(images[i], axis=0),
                         y_label: np.expand_dims(labels[i], axis=0)}
          acc_val = sess.run(acc, feed_dict=feed_dict_i)
          tf.logging.info('Accuracy: %s', acc_val)
          num_correct += acc_val
        assert (num_correct / num_images) < 0.1


if __name__ == '__main__':
  unittest.main()
