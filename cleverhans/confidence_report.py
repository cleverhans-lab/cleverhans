"""Functionality for making confidence reports.

A confidence report is a dictionary.
Each dictionary key is the name of a type of data:
  clean : Clean data
  bundled : bundled adversarial examples
Each value in the dictionary contains an array of bools indicating whether
the model got each example correct and an array containing the confidence
that the model assigned to each prediction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

from cleverhans.utils import set_log_level
from cleverhans.serial import load

# Defaults. Imported elsewhere so that command line script defaults match
# function defaults.
TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
WHICH_SET = 'test'

def make_confidence_report_bundled(filepath, train_start=TRAIN_START,
                                   train_end=TRAIN_END, test_start=TEST_START,
                                   test_end=TEST_END, which_set=WHICH_SET):
  """
  Load a saved model, gather its predictions, and save a confidence report.
  :param filepath: path to model to evaluate
  :param train_start: index of first training set example to use
  :param train_end: index of last training set example to use
  :param test_start: index of first test set example to use
  :param test_end: index of last test set example to use
  :param which_set: 'train' or 'test'
  """
  # Avoid circular import
  from cleverhans.attack_bundling import basic_max_confidence_recipe

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.INFO)

  # Create TF session
  sess = tf.Session()

  assert filepath.endswith('.joblib')
  report_path = filepath[:-len('.joblib')] + "_bundled_report.joblib"

  with sess.as_default():
    model = load(filepath)
  assert len(model.get_params()) > 0
  factory = model.dataset_factory
  factory.kwargs['test_start'] = test_start
  factory.kwargs['test_end'] = test_end
  dataset = factory()

  center = factory.kwargs['center']
  value_range = 1. + center
  min_value = 0. - center
  max_value = 1.

  if 'CIFAR' in str(factory.cls):
    base_eps = 8. / 255.
    base_eps_iter = 2. / 255.
  elif 'MNIST' in str(factory.cls):
    base_eps = .3
    base_eps_iter = .1
  else:
    raise NotImplementedError(str(factory.cls))

  eps = base_eps * value_range
  eps_iter = base_eps_iter * value_range
  nb_iter = 40
  clip_min = min_value
  clip_max = max_value

  x_data, y_data = dataset.get_set(which_set)
  assert x_data.max() <= max_value
  assert x_data.min() >= min_value

  basic_max_confidence_recipe(sess, model=model, x=x_data, y=y_data,
                              nb_classes=dataset.NB_CLASSES, eps=eps,
                              clip_min=clip_min, clip_max=clip_max,
                              eps_iter=eps_iter, nb_iter=nb_iter,
                              report_path=report_path)

def print_stats(correctness, confidence, name):
  """
  Prints out accuracy, coverage, etc. statistics
  :param correctness: ndarray
    One bool per example specifying whether it was correctly classified
  :param confidence: ndarray
    The probability associated with each prediction
  :param name: str
    The name of this type of data (e.g. "clean", "MaxConfidence")
  """
  accuracy = correctness.mean()
  wrongness = 1 - correctness
  denom1 = np.maximum(1, wrongness.sum())
  ave_prob_on_mistake = (wrongness * confidence).sum() / denom1
  assert ave_prob_on_mistake <= 1., ave_prob_on_mistake
  denom2 = np.maximum(1, correctness.sum())
  ave_prob_on_correct = (correctness * confidence).sum() / denom2
  covered = confidence > 0.5
  cov_half = covered.mean()
  acc_half = (correctness * covered).sum() / np.maximum(1, covered.sum())
  print('Accuracy on %s examples: %0.4f' % (name, accuracy))
  print("Average prob on mistakes: %0.4f" % ave_prob_on_mistake)
  print("Average prob on correct: %0.4f" % ave_prob_on_correct)
  print("Accuracy when prob thresholded at .5: %0.4f" % acc_half)
  print("Coverage when prob thresholded at .5: %0.4f" % cov_half)

  success_rate = acc_half * cov_half
  # Success is correctly classifying a covered example
  print("Success rate at .5: %0.4f" % success_rate)
  # Failure is misclassifying a covered example
  failure_rate = (1. - acc_half) * cov_half
  print("Failure rate at .5: %0.4f" % failure_rate)
  print()
