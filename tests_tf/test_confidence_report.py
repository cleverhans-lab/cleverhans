"""
Tests for cleverhans.confidence_report
"""

import numpy as np
import tensorflow as tf

from cleverhans.attacks import Noise
from cleverhans.attack_bundling import AttackConfig
from cleverhans.attack_bundling import bundle_attacks
from cleverhans.attack_bundling import Misclassify
from cleverhans.confidence_report import ConfidenceReport
from cleverhans.confidence_report import ConfidenceReportEntry
from cleverhans.confidence_report import make_confidence_report_bundled
from cleverhans.devtools.mocks import SimpleDataset
from cleverhans.picklable_model import MLP, Linear
from cleverhans import serial


def test_confidence_report():
  """
  Test that we can make a confidence report, put an entry in it, and get
  that entry back out
  """
  report = ConfidenceReport()
  entry = ConfidenceReportEntry(correctness=np.array([True, False]),
                                confidence=np.array([0.9, 0.1]))
  report['clean'] = entry
  assert report['clean'] is entry


def test_make_confidence_report_bundled():
  """
  A very simple test that just makes sure make_confidence_report_bundled can run without crashing
  """

  sess = tf.Session()
  try:
    nb_classes = 3
    nb_features = 2
    batch_size = 5
    nb_test_examples = batch_size * 2
    layer = Linear(num_hid=nb_classes)
    model = MLP(layers=[layer], input_shape=(None, nb_features))
    dataset = SimpleDataset(test_end=nb_test_examples, nb_classes=nb_classes)
    model.dataset_factory = dataset.get_factory()
    filepath = ".test_model.joblib"
    with sess.as_default():
      sess.run(tf.global_variables_initializer())
      serial.save(filepath, model)
    def recipe(sess, model, x, y, nb_classes, eps, clip_min,
               clip_max, eps_iter, nb_iter,
               report_path, eps_iter_small, batch_size):
      """
      Mock recipe that just runs the Noise attack so the test runs fast
      """
      attack_configs = [AttackConfig(Noise(model, sess), {'eps': eps})]
      new_work_goal = {config: 1 for config in attack_configs}
      goals = [Misclassify(new_work_goal=new_work_goal)]
      bundle_attacks(sess, model, x, y, attack_configs, goals, report_path, attack_batch_size=batch_size,
                     eval_batch_size=batch_size)
    make_confidence_report_bundled(filepath, test_end=nb_test_examples, recipe=recipe,
                                   base_eps=.1, base_eps_iter=.01, batch_size=batch_size)
  finally:
    sess.close()

def test_save_load_confidence_report():
  """
  Test that a confidence report can be loaded and saved.
  """
  report = ConfidenceReport()
  num_examples = 2
  clean_correctness = np.zeros((num_examples,), dtype=np.bool)
  clean_confidence = np.zeros((num_examples,), dtype=np.float32)
  adv_correctness = clean_correctness.copy()
  adv_confidence = clean_confidence.copy()
  report['clean'] = ConfidenceReportEntry(clean_correctness, clean_confidence)
  report['adv'] = ConfidenceReportEntry(adv_correctness, adv_confidence)
  report.completed = True
  filepath = ".test_confidence_report.joblib"
  serial.save(filepath, report)
  report = serial.load(filepath)
