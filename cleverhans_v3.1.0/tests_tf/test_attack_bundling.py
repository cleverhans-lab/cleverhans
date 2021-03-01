"""
Tests for cleverhans.attack_bundling
"""
import numpy as np
from cleverhans.attack_bundling import AttackConfig
from cleverhans.attack_bundling import Misclassify
from cleverhans.attack_bundling import unfinished_attack_configs


def test_unfinished_attack_configs():
  """
  Test that tracking of which attack configs are unfinished is correct
  """

  new_work_goal = {}
  work_before = {}
  run_counts = {}

  expected_unfinished = []
  expected_finished = []

  easy_finished = AttackConfig(None, None)
  new_work_goal[easy_finished] = 1
  work_before[easy_finished] = np.array([0, 0])
  run_counts[easy_finished] = np.array([1, 1])
  expected_finished.append(easy_finished)

  easy_unfinished = AttackConfig(None, None)
  new_work_goal[easy_unfinished] = 1
  work_before[easy_unfinished] = np.array([0, 0])
  run_counts[easy_unfinished] = np.array([0, 0])
  expected_unfinished.append(easy_unfinished)

  only_partly_finished = AttackConfig(None, None)
  new_work_goal[only_partly_finished] = 1
  work_before[only_partly_finished] = np.array([0, 0])
  run_counts[only_partly_finished] = np.array([1, 0])
  expected_unfinished.append(only_partly_finished)

  work_not_new = AttackConfig(None, None)
  new_work_goal[work_not_new] = 1
  work_before[work_not_new] = np.array([1, 1])
  run_counts[work_not_new] = np.array([1, 1])
  expected_unfinished.append(work_not_new)

  actual_unfinished = unfinished_attack_configs(new_work_goal, work_before,
                                                run_counts)

  assert all(e in actual_unfinished for e in expected_unfinished)
  assert all(e not in actual_unfinished for e in expected_finished)


def test_misclassify_request_examples():
  """
  Test Misclassify.request_examples
  """
  cfg = AttackConfig(None, None)
  goal = Misclassify(new_work_goal={cfg: 1})
  correctness = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool)
  run_counts = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int64)
  criteria = {'correctness': correctness}
  batch_size = 3
  idxs = goal.request_examples(cfg, criteria, {cfg: run_counts}, batch_size)
  assert idxs.shape == (batch_size,)
  idxs = list(idxs)
  for already_misclassified in [0, 2, 4, 6, 8]:
    assert already_misclassified not in idxs
  for already_run in [1, 7]:
    assert already_run not in idxs
  for needed in [3, 5, 9]:
    assert needed in idxs
