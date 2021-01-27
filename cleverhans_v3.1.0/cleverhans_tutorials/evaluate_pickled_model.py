"""
This script evaluates trained models that have been saved to the filesystem.
See mnist_tutorial_picklable.py for instructions.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import tensorflow as tf

from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, silence
from cleverhans.serial import load
silence()

FLAGS = flags.FLAGS


def evaluate_model(filepath,
                   train_start=0, train_end=60000, test_start=0,
                   test_end=10000, batch_size=128,
                   testing=False, num_threads=None):
  """
  Run evaluation on a saved model
  :param filepath: path to model to evaluate
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param batch_size: size of evaluation batches
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.INFO)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get MNIST test data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  eval_params = {'batch_size': batch_size}
  fgsm_params = {
      'eps': 0.3,
      'clip_min': 0.,
      'clip_max': 1.
  }

  def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  with sess.as_default():
    model = load(filepath)
  assert len(model.get_params()) > 0

  # Initialize the Fast Gradient Sign Method (FGSM) attack object and
  # graph
  fgsm = FastGradientMethod(model, sess=sess)
  adv_x = fgsm.generate(x, **fgsm_params)
  preds_adv = model.get_logits(adv_x)
  preds = model.get_logits(x)

  # Evaluate the accuracy of the MNIST model on adversarial examples
  do_eval(preds, x_test, y_test, 'train_clean_train_clean_eval', False)
  do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)


def main(argv=None):
  _, filepath = argv
  evaluate_model(filepath=filepath)


if __name__ == '__main__':
  tf.app.run()
