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

from collections import OrderedDict
import logging
import time
import warnings

import numpy as np
import six
import tensorflow as tf

from cleverhans.attacks import MaxConfidence
from cleverhans.attacks import Semantic
from cleverhans.evaluation import correctness_and_confidence
from cleverhans.evaluation import run_attack
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
BASE_EPS_ITER = None  # Differs by dataset
SAVE_ADVX = 1


class ConfidenceReport(OrderedDict):
  """
  A data structure reporting how much confidence a model assigned to its
  predictions on each example and whether those predictions were correct.
  This class is just a dictionary with some type checks.
  It maps string data type names (like "clean" for clean data or "Semantic"
  for semantic adversarial examples) to ConfidenceReportEntry instances.

  :param iterable: optional iterable containing (key, value) tuples
  """

  def __init__(self, iterable=None):
    super(ConfidenceReport, self).__init__()
    # This field tracks whether the report is completed.
    # It's important e.g. for reports that are made by bundlers and repeatedly
    # written to disk during the process. This field makes it possible to tell
    # whether a report on disk is complete or whether the bundling process
    # got killed (e.g. due to VM migration)
    self.completed = False
    if iterable is not None:
      # pickle sometimes wants to use this interface to unpickle the OrderedDict
      for key, value in iterable:
        self[key] = value

  def __setitem__(self, key, value):
    assert isinstance(key, six.string_types)
    if not isinstance(value, ConfidenceReportEntry):
      raise TypeError("`value` must be a ConfidenceReportEntry, but got " + str(value) + " of type " + str(type(value)))
    super(ConfidenceReport, self).__setitem__(key, value)

class ConfidenceReportEntry(object):
  """
  A data structure reporting how much confidence a model assigned to its
  predictions on each example and whether those predictions were correct.

  :param correctness: ndarray, one bool per example indicating whether it was
    correct
  :param confidence: ndarray, one floating point value per example reporting
    the probability assigned to the prediction for that example
  """
  def __init__(self, correctness, confidence):
    assert isinstance(correctness, np.ndarray)
    assert isinstance(correctness, np.ndarray)
    assert correctness.ndim == 1
    assert confidence.ndim == 1
    assert correctness.dtype == np.bool, correctness.dtype
    assert np.issubdtype(confidence.dtype, np.floating)
    assert correctness.shape == confidence.shape
    assert confidence.min() >= 0.
    assert confidence.max() <= 1.
    self.correctness = correctness
    self.confidence = confidence

  def __getitem__(self, key):
    warnings.warn("Dictionary confidence report entries are deprecated. "
                  "Switch to accessing the appropriate field of "
                  "ConfidenceReportEntry. "
                  "Dictionary-style access will be removed on or after "
                  "2019-04-24.")
    assert key in ['correctness', 'confidence']
    return self.__dict__[key]

  def __setitem__(self, key, value):
    warnings.warn("Dictionary confidence report entries are deprecated."
                  "Switch to accessing the appropriate field of "
                  "ConfidenceReportEntry. "
                  "Dictionary-style access will be removed on or after "
                  "2019-04-24.")
    assert key in ['correctness', 'confidence']
    self.__dict__[key] = value


def make_confidence_report_bundled(filepath, train_start=TRAIN_START,
                                   train_end=TRAIN_END, test_start=TEST_START,
                                   test_end=TEST_END, which_set=WHICH_SET,
                                   recipe=RECIPE, report_path=REPORT_PATH,
                                   nb_iter=NB_ITER, base_eps=None,
                                   base_eps_iter=None, base_eps_iter_small=None,
                                   batch_size=BATCH_SIZE):
  """
  Load a saved model, gather its predictions, and save a confidence report.
  :param filepath: path to model to evaluate
  :param train_start: index of first training set example to use
  :param train_end: index of last training set example to use
  :param test_start: index of first test set example to use
  :param test_end: index of last test set example to use
  :param which_set: 'train' or 'test'
  :param nb_iter: int, number of iterations of attack algorithm
    (note that different recipes will use this differently,
     for example many will run two attacks, one with nb_iter
     iterations and one with 25X more)
  :param base_eps: float, epsilon parameter for threat model, on a scale of [0, 1].
    Inferred from the dataset if not specified.
  :param base_eps_iter: float, a step size used in different ways by different recipes.
    Typically the step size for a PGD attack.
    Inferred from the dataset if not specified.
  :param base_eps_iter_small: float, a second step size for a more fine-grained attack.
    Inferred from the dataset if not specified.
  :param batch_size: int, batch size
  """
  # Avoid circular import
  from cleverhans import attack_bundling
  if callable(recipe):
    run_recipe = recipe
  else:
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
  factory.kwargs['train_start'] = train_start
  factory.kwargs['train_end'] = train_end
  factory.kwargs['test_start'] = test_start
  factory.kwargs['test_end'] = test_end
  dataset = factory()

  center = dataset.kwargs['center']
  if 'max_val' in factory.kwargs:
    max_value = factory.kwargs['max_val']
  elif hasattr(dataset, 'max_val'):
    max_value = dataset.max_val
  else:
    raise AttributeError("Can't find max_value specification")
  min_value = 0. - center * max_value
  value_range = max_value - min_value

  if 'CIFAR' in str(factory.cls):
    if base_eps is None:
      base_eps = 8. / 255.
    if base_eps_iter is None:
      base_eps_iter = 2. / 255.
    if base_eps_iter_small is None:
      base_eps_iter_small = 1. / 255.
  elif 'MNIST' in str(factory.cls):
    if base_eps is None:
      base_eps = .3
    if base_eps_iter is None:
      base_eps_iter = .1
    base_eps_iter_small = None
  else:
    # Note that it is not required to specify base_eps_iter_small
    if base_eps is None or base_eps_iter is None:
      raise NotImplementedError("Not able to infer threat model from " + str(factory.cls))

  eps = base_eps * value_range
  eps_iter = base_eps_iter * value_range
  if base_eps_iter_small is None:
    eps_iter_small = None
  else:
    eps_iter_small = base_eps_iter_small * value_range
  clip_min = min_value
  clip_max = max_value

  x_data, y_data = dataset.get_set(which_set)
  assert x_data.max() <= max_value
  assert x_data.min() >= min_value

  assert eps_iter <= eps
  assert eps_iter_small is None or eps_iter_small <= eps

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
               report_path=report_path, eps_iter_small=eps_iter_small, batch_size=batch_size)


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


def make_confidence_report(filepath, train_start=TRAIN_START,
                           train_end=TRAIN_END,
                           test_start=TEST_START, test_end=TEST_END,
                           batch_size=BATCH_SIZE, which_set=WHICH_SET,
                           mc_batch_size=MC_BATCH_SIZE,
                           report_path=REPORT_PATH,
                           base_eps_iter=BASE_EPS_ITER,
                           nb_iter=NB_ITER, save_advx=SAVE_ADVX):
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
  :param save_advx: bool. If True, saves the adversarial examples to disk.
    On by default, but can be turned off to save memory, etc.
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

  report = ConfidenceReport()

  semantic = Semantic(model, center, max_val, sess)
  mc = MaxConfidence(model, sess=sess)

  jobs = [('clean', None, None, None, False),
          ('Semantic', semantic, None, None, False),
          ('mc', mc, mc_params, mc_batch_size, True)]

  for job in jobs:
    name, attack, attack_params, job_batch_size, save_this_job = job
    if job_batch_size is None:
      job_batch_size = batch_size
    t1 = time.time()
    if save_advx and save_this_job:
      # If we want to save the adversarial examples to the filesystem, we need
      # to fetch all of them. Otherwise they're just computed one batch at a
      # time and discarded

      # The path to save to
      assert report_path.endswith('.joblib')
      advx_path = report_path[:-len('.joblib')] + '_advx_' + name + '.npy'

      # Fetch the adversarial examples
      x_data = run_attack(sess, model, x_data, y_data, attack, attack_params,
                          batch_size=job_batch_size, devices=devices)

      # Turn off the attack so `correctness_and_confidence` won't run it a
      # second time.
      attack = None
      attack_params = None

      # Save the adversarial examples
      np.save(advx_path, x_data)

    # Run correctness and confidence evaluation on adversarial examples
    packed = correctness_and_confidence(sess, model, x_data, y_data,
                                        batch_size=job_batch_size,
                                        devices=devices,
                                        attack=attack,
                                        attack_params=attack_params)
    t2 = time.time()
    print("Evaluation took", t2 - t1, "seconds")
    correctness, confidence = packed

    report[name] = ConfidenceReportEntry(correctness=correctness,
                                         confidence=confidence)

    print_stats(correctness, confidence, name)

  save(report_path, report)
