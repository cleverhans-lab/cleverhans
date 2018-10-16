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
import time

import numpy as np
import tensorflow as tf

from cleverhans.attacks import MaxConfidence
from cleverhans.attacks import Semantic
from cleverhans.evaluation import correctness_and_confidence
from cleverhans.utils import set_log_level
from cleverhans.serial import load, save
from cleverhans.utils_tf import infer_devices

# Defaults. Imported elsewhere so that command line script defaults match
# function defaults.
TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
WHICH_SET = 'test'
RECIPE = 'basic_max_confidence_recipe'
REPORT_PATH = None
# Used for `make_confidence_report` but not `make_confidence_report_bundled`
devices = infer_devices()
num_devices = len(devices)
BATCH_SIZE = 128 * num_devices
MC_BATCH_SIZE = 16 * num_devices
NB_ITER = 40
BASE_EPS_ITER = None # Differs by dataset

def make_confidence_report_bundled(filepath, train_start=TRAIN_START,
                                   train_end=TRAIN_END, test_start=TEST_START,
                                   test_end=TEST_END, which_set=WHICH_SET,
                                   recipe=RECIPE, report_path=REPORT_PATH):
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
  from cleverhans import attack_bundling
  run_recipe = getattr(attack_bundling, recipe)

  # Set logging level to see debug information
  set_log_level(logging.INFO)

  # Create TF session
  sess = tf.Session()

  assert filepath.endswith('.joblib')
  if report_path is None:
    report_path = filepath[:-len('.joblib')] + "_bundled_report.joblib"

  with sess.as_default():
    model = load(filepath)
  assert len(model.get_params()) > 0
  factory = model.dataset_factory
  factory.kwargs['test_start'] = test_start
  factory.kwargs['test_end'] = test_end
  dataset = factory()

  center = dataset.kwargs['center']
  max_value = dataset.max_val
  min_value = 0. - center * max_value
  value_range = max_value - min_value

  if 'CIFAR' in str(factory.cls):
    base_eps = 8. / 255.
    base_eps_iter = 2. / 255.
    base_eps_iter_small = 1. / 255.
  elif 'MNIST' in str(factory.cls):
    base_eps = .3
    base_eps_iter = .1
    base_eps_iter_small = None
  else:
    raise NotImplementedError(str(factory.cls))

  eps = base_eps * value_range
  eps_iter = base_eps_iter * value_range
  if base_eps_iter_small is None:
    eps_iter_small = None
  else:
    eps_iter_small = base_eps_iter_small * value_range
  nb_iter = 40
  clip_min = min_value
  clip_max = max_value

  x_data, y_data = dataset.get_set(which_set)
  assert x_data.max() <= max_value
  assert x_data.min() >= min_value

  assert eps_iter <= eps
  assert eps_iter_small <= eps

  # Different recipes take different arguments.
  # For now I don't have an idea for a beautiful unifying framework, so
  # we get an if statement.
  if recipe == 'random_search_max_confidence_recipe':
    # pylint always checks against the default recipe here
    # pylint: disable=no-value-for-parameter
    run_recipe(sess=sess, model=model, x=x_data, y=y_data, eps=eps,
               clip_min=clip_min, clip_max=clip_max, report_path=report_path)
  else:
    run_recipe(sess=sess, model=model, x=x_data, y=y_data,
               nb_classes=dataset.NB_CLASSES, eps=eps, clip_min=clip_min,
               clip_max=clip_max, eps_iter=eps_iter, nb_iter=nb_iter,
               report_path=report_path, eps_iter_small=eps_iter_small)



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

def make_confidence_report(filepath, train_start=TRAIN_START, train_end=TRAIN_END,
                           test_start=TEST_START, test_end=TEST_END,
                           batch_size=BATCH_SIZE, which_set=WHICH_SET,
                           mc_batch_size=MC_BATCH_SIZE,
                           report_path=REPORT_PATH,
                           base_eps_iter=BASE_EPS_ITER,
                           nb_iter=NB_ITER):
  """
  Load a saved model, gather its predictions, and save a confidence report.


  This function works by running a single MaxConfidence attack on each example.
  This provides a reasonable estimate of the true failure rate quickly, so
  long as the model does not suffer from gradient masking.
  However, this estimate is mostly intended for development work and not
  for publication. A more accurate estimate may be obtained by running
  make_confidence_report_bundled.py instead.

  :param filepath: path to model to evaluate
  :param train_start: index of first training set example to use
  :param train_end: index of last training set example to use
  :param test_start: index of first test set example to use
  :param test_end: index of last test set example to use
  :param batch_size: size of evaluation batches
  :param which_set: 'train' or 'test'
  :param mc_batch_size: batch size for MaxConfidence attack
  :param base_eps_iter: step size if the data were in [0,1]
    (Step size will be rescaled proportional to the actual data range)
  :param nb_iter: Number of iterations of PGD to run per class
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.INFO)

  # Create TF session
  sess = tf.Session()

  if report_path is None:
    assert filepath.endswith('.joblib')
    report_path = filepath[:-len('.joblib')] + "_report.joblib"

  with sess.as_default():
    model = load(filepath)
  assert len(model.get_params()) > 0
  factory = model.dataset_factory
  factory.kwargs['train_start'] = train_start
  factory.kwargs['train_end'] = train_end
  factory.kwargs['test_start'] = test_start
  factory.kwargs['test_end'] = test_end
  dataset = factory()

  center = dataset.kwargs['center']
  max_val = dataset.kwargs['max_val']
  value_range = max_val * (1. + center)
  min_value = 0. - center * max_val

  if 'CIFAR' in str(factory.cls):
    base_eps = 8. / 255.
    if base_eps_iter is None:
      base_eps_iter = 2. / 255.
  elif 'MNIST' in str(factory.cls):
    base_eps = .3
    if base_eps_iter is None:
      base_eps_iter = .1
  else:
    raise NotImplementedError(str(factory.cls))

  mc_params = {'eps': base_eps * value_range,
               'eps_iter': base_eps_iter * value_range,
               'nb_iter': nb_iter,
               'clip_min': min_value,
               'clip_max': max_val}


  x_data, y_data = dataset.get_set(which_set)

  report = {}

  semantic = Semantic(model, center, max_val, sess)
  mc = MaxConfidence(model, sess=sess)

  jobs = [('clean', None, None, None),
          ('Semantic', semantic, None, None),
          ('mc', mc, mc_params, mc_batch_size)]


  for job in jobs:
    name, attack, attack_params, job_batch_size = job
    if job_batch_size is None:
      job_batch_size = batch_size
    t1 = time.time()
    packed = correctness_and_confidence(sess, model, x_data, y_data,
                                        batch_size=job_batch_size, devices=devices,
                                        attack=attack,
                                        attack_params=attack_params)
    t2 = time.time()
    print("Evaluation took", t2 - t1, "seconds")
    correctness, confidence = packed

    report[name] = {
        'correctness' : correctness,
        'confidence'  : confidence
        }

    print_stats(correctness, confidence, name)


  save(report_path, report)
