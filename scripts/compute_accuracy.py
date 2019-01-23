#!/usr/bin/env python3
"""
compute_accuracy.py
Usage:
  compute_accuracy.py model.joblib

  where model.joblib is a file created by cleverhans.serial.save containing
  a picklable cleverhans.model.Model instance.

This script will run the model on a variety of types of data and print out
the accuracy for each data type.
  clean : Clean data
  semantic : Semantic adversarial examples
  pgd: PGD adversarial examples

This script works by running a single attack on each example.
This is useful for quick evaluation during development, but for final
publication it would be better to use attack bundling.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import time

import tensorflow as tf

from cleverhans.attacks import ProjectedGradientDescent, Semantic
from cleverhans.compat import flags
from cleverhans.evaluation import accuracy
from cleverhans.serial import load
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import infer_devices
from cleverhans.utils_tf import silence
silence()
devices = infer_devices()
num_devices = len(devices)
BATCH_SIZE = 128
TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
WHICH_SET = 'test'
NB_ITER = 40
BASE_EPS_ITER = None  # Differs by dataset


FLAGS = flags.FLAGS


def print_accuracies(filepath, train_start=TRAIN_START, train_end=TRAIN_END,
                     test_start=TEST_START, test_end=TEST_END,
                     batch_size=BATCH_SIZE, which_set=WHICH_SET,
                     base_eps_iter=BASE_EPS_ITER,
                     nb_iter=NB_ITER):
  """
  Load a saved model and print out its accuracy on different data distributions

  This function works by running a single attack on each example.
  This provides a reasonable estimate of the true failure rate quickly, so
  long as the model does not suffer from gradient masking.
  However, this estimate is mostly intended for development work and not
  for publication. A more accurate estimate may be obtained by running
  an attack bundler instead.

  :param filepath: path to model to evaluate
  :param train_start: index of first training set example to use
  :param train_end: index of last training set example to use
  :param test_start: index of first test set example to use
  :param test_end: index of last test set example to use
  :param batch_size: size of evaluation batches
  :param which_set: 'train' or 'test'
  :param base_eps_iter: step size if the data were in [0,1]
    (Step size will be rescaled proportional to the actual data range)
  :param nb_iter: Number of iterations of PGD to run per class
  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(20181014)
  set_log_level(logging.INFO)
  sess = tf.Session()

  with sess.as_default():
    model = load(filepath)
  assert len(model.get_params()) > 0
  factory = model.dataset_factory
  factory.kwargs['train_start'] = train_start
  factory.kwargs['train_end'] = train_end
  factory.kwargs['test_start'] = test_start
  factory.kwargs['test_end'] = test_end
  dataset = factory()


  x_data, y_data = dataset.get_set(which_set)

  impl(sess, model, dataset, factory, x_data, y_data, base_eps_iter, nb_iter)

def impl(sess, model, dataset, factory, x_data, y_data,
         base_eps_iter=BASE_EPS_ITER, nb_iter=NB_ITER,
         batch_size=BATCH_SIZE):
  """
  The actual implementation of the evaluation.
  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param dataset: cleverhans.dataset.Dataset
  :param factory: the dataset factory corresponding to `dataset`
  :param x_data: numpy array of input examples
  :param y_data: numpy array of class labels
  :param base_eps_iter: step size for PGD if data were in [0, 1]
  :param nb_iter: number of PGD iterations
  :returns: dict mapping string adversarial example names to accuracies
  """

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

  pgd_params = {'eps': base_eps * value_range,
                'eps_iter': base_eps_iter * value_range,
                'nb_iter': nb_iter,
                'clip_min': min_value,
                'clip_max': max_val}

  semantic = Semantic(model, center, max_val, sess)
  pgd = ProjectedGradientDescent(model, sess=sess)

  jobs = [('clean', None, None, None),
          ('Semantic', semantic, None, None),
          ('pgd', pgd, pgd_params, None)]

  out = {}

  for job in jobs:
    name, attack, attack_params, job_batch_size = job
    if job_batch_size is None:
      job_batch_size = batch_size
    t1 = time.time()
    acc = accuracy(sess, model, x_data, y_data, batch_size=job_batch_size,
                   devices=devices, attack=attack, attack_params=attack_params)
    t2 = time.time()
    out[name] = acc
    print("Accuracy on " + name + " examples: ", acc)
    print("Evaluation took", t2 - t1, "seconds")

  return out


def main(argv=None):
  """
  Print accuracies
  """
  try:
    _name_of_script, filepath = argv
  except ValueError:
    raise ValueError(argv)
  print_accuracies(filepath=filepath, test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end, which_set=FLAGS.which_set,
                   nb_iter=FLAGS.nb_iter, base_eps_iter=FLAGS.base_eps_iter,
                   batch_size=FLAGS.batch_size)


if __name__ == '__main__':
  flags.DEFINE_integer('train_start', TRAIN_START, 'Starting point (inclusive)'
                       'of range of train examples to use')
  flags.DEFINE_integer('train_end', TRAIN_END, 'Ending point (non-inclusive) '
                       'of range of train examples to use')
  flags.DEFINE_integer('test_start', TEST_START, 'Starting point (inclusive) '
                       'of range of test examples to use')
  flags.DEFINE_integer('test_end', TEST_END, 'End point (non-inclusive) of '
                       'range of test examples to use')
  flags.DEFINE_integer('nb_iter', NB_ITER, 'Number of iterations of PGD')
  flags.DEFINE_string('which_set', WHICH_SET, '"train" or "test"')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Batch size for most jobs')
  flags.DEFINE_float('base_eps_iter', BASE_EPS_ITER,
                     'epsilon per iteration, if data were in [0, 1]')
  tf.app.run()
