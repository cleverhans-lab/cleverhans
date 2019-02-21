"""
Runs multiple attacks against each example.

References: https://openreview.net/forum?id=H1g0piA9tQ
            https://arxiv.org/abs/1811.03685
"""
# pylint: disable=missing-docstring
import copy
import logging
import time

import numpy as np
import six
from six.moves import range
import tensorflow as tf

from cleverhans.attacks import Noise
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import SPSA
from cleverhans.evaluation import correctness_and_confidence
from cleverhans.evaluation import batch_eval_multi_worker, run_attack
from cleverhans.model import Model
from cleverhans import serial
from cleverhans.utils import create_logger, deep_copy, safe_zip
from cleverhans.utils_tf import infer_devices
from cleverhans.confidence_report import ConfidenceReport
from cleverhans.confidence_report import ConfidenceReportEntry
from cleverhans.confidence_report import print_stats

_logger = create_logger("attack_bundling")
_logger.setLevel(logging.INFO)

devices = infer_devices()
num_devices = len(devices)
DEFAULT_EXAMPLES_PER_DEVICE = 128
BATCH_SIZE = DEFAULT_EXAMPLES_PER_DEVICE * num_devices
REPORT_TIME_INTERVAL = 60

# TODO: lower priority: make it possible to initialize one attack with
# the output of an earlier attack


def single_run_max_confidence_recipe(sess, model, x, y, nb_classes, eps,
                                     clip_min, clip_max, eps_iter, nb_iter,
                                     report_path,
                                     batch_size=BATCH_SIZE,
                                     eps_iter_small=None):
  """A reasonable attack bundling recipe for a max norm threat model and
  a defender that uses confidence thresholding. This recipe uses both
  uniform noise and randomly-initialized PGD targeted attacks.

  References:
  https://openreview.net/forum?id=H1g0piA9tQ

  This version runs each attack (noise, targeted PGD for each class with
  nb_iter iterations, target PGD for each class with 25X more iterations)
  just once and then stops. See `basic_max_confidence_recipe` for a version
  that runs indefinitely.

  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param nb_classes: int, number of classes
  :param eps: float, maximum size of perturbation (measured by max norm)
  :param eps_iter: float, step size for one version of PGD attacks
    (will also run another version with eps_iter_small step size)
  :param nb_iter: int, number of iterations for the cheaper PGD attacks
    (will also run another version with 25X more iterations)
  :param report_path: str, the path that the report will be saved to.
  :param batch_size: int, the total number of examples to run simultaneously
  :param eps_iter_small: optional, float.
    The second version of the PGD attack is run with 25 * nb_iter iterations
    and eps_iter_small step size. If eps_iter_small is not specified it is
    set to eps_iter / 25.
  """
  noise_attack = Noise(model, sess)
  pgd_attack = ProjectedGradientDescent(model, sess)
  threat_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max}
  noise_attack_config = AttackConfig(noise_attack, threat_params, "noise")
  attack_configs = [noise_attack_config]
  pgd_attack_configs = []
  pgd_params = copy.copy(threat_params)
  pgd_params["eps_iter"] = eps_iter
  pgd_params["nb_iter"] = nb_iter
  assert batch_size % num_devices == 0
  dev_batch_size = batch_size // num_devices
  ones = tf.ones(dev_batch_size, tf.int32)
  expensive_pgd = []
  if eps_iter_small is None:
    eps_iter_small = eps_iter / 25.
  for cls in range(nb_classes):
    cls_params = copy.copy(pgd_params)
    cls_params['y_target'] = tf.to_float(tf.one_hot(ones * cls, nb_classes))
    cls_attack_config = AttackConfig(pgd_attack, cls_params, "pgd_" + str(cls))
    pgd_attack_configs.append(cls_attack_config)
    expensive_params = copy.copy(cls_params)
    expensive_params["eps_iter"] = eps_iter_small
    expensive_params["nb_iter"] *= 25.
    expensive_config = AttackConfig(
        pgd_attack, expensive_params, "expensive_pgd_" + str(cls))
    expensive_pgd.append(expensive_config)
  attack_configs = [noise_attack_config] + pgd_attack_configs + expensive_pgd
  new_work_goal = {config: 1 for config in attack_configs}
  goals = [MaxConfidence(t=1., new_work_goal=new_work_goal)]
  bundle_attacks(sess, model, x, y, attack_configs, goals, report_path, attack_batch_size=batch_size,
                 eval_batch_size=batch_size)


def basic_max_confidence_recipe(sess, model, x, y, nb_classes, eps,
                                clip_min, clip_max, eps_iter, nb_iter,
                                report_path,
                                batch_size=BATCH_SIZE,
                                eps_iter_small=None):
  """A reasonable attack bundling recipe for a max norm threat model and
  a defender that uses confidence thresholding.

  References:
  https://openreview.net/forum?id=H1g0piA9tQ

  This version runs indefinitely, updating the report on disk continuously.

  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param nb_classes: int, number of classes
  :param eps: float, maximum size of perturbation (measured by max norm)
  :param eps_iter: float, step size for one version of PGD attacks
    (will also run another version with eps_iter_small)
  :param nb_iter: int, number of iterations for one version of PGD attacks
    (will also run another version with 25X more iterations)
  :param report_path: str, the path that the report will be saved to.
  :batch_size: int, the total number of examples to run simultaneously
  :param eps_iter_small: optional, float.
    The second version of the PGD attack is run with 25 * nb_iter iterations
    and eps_iter_small step size. If eps_iter_small is not specified it is
    set to eps_iter / 25.
  """
  noise_attack = Noise(model, sess)
  pgd_attack = ProjectedGradientDescent(model, sess)
  threat_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max}
  noise_attack_config = AttackConfig(noise_attack, threat_params)
  attack_configs = [noise_attack_config]
  pgd_attack_configs = []
  pgd_params = copy.copy(threat_params)
  pgd_params["eps_iter"] = eps_iter
  pgd_params["nb_iter"] = nb_iter
  assert batch_size % num_devices == 0
  dev_batch_size = batch_size // num_devices
  ones = tf.ones(dev_batch_size, tf.int32)
  expensive_pgd = []
  if eps_iter_small is None:
    eps_iter_small = eps_iter / 25.
  for cls in range(nb_classes):
    cls_params = copy.copy(pgd_params)
    cls_params['y_target'] = tf.to_float(tf.one_hot(ones * cls, nb_classes))
    cls_attack_config = AttackConfig(pgd_attack, cls_params, "pgd_" + str(cls))
    pgd_attack_configs.append(cls_attack_config)
    expensive_params = copy.copy(cls_params)
    expensive_params["eps_iter"] = eps_iter_small
    expensive_params["nb_iter"] *= 25.
    expensive_config = AttackConfig(
        pgd_attack, expensive_params, "expensive_pgd_" + str(cls))
    expensive_pgd.append(expensive_config)
  attack_configs = [noise_attack_config] + pgd_attack_configs + expensive_pgd
  new_work_goal = {config: 5 for config in attack_configs}
  pgd_work_goal = {config: 5 for config in pgd_attack_configs}
  goals = [Misclassify(new_work_goal={noise_attack_config: 50}),
           Misclassify(new_work_goal=pgd_work_goal),
           MaxConfidence(t=0.5, new_work_goal=new_work_goal),
           MaxConfidence(t=0.75, new_work_goal=new_work_goal),
           MaxConfidence(t=0.875, new_work_goal=new_work_goal),
           MaxConfidence(t=0.9375, new_work_goal=new_work_goal),
           MaxConfidence(t=0.96875, new_work_goal=new_work_goal),
           MaxConfidence(t=0.984375, new_work_goal=new_work_goal),
           MaxConfidence(t=1.)]
  bundle_attacks(sess, model, x, y, attack_configs, goals, report_path)
  # This runs forever


def fixed_max_confidence_recipe(sess, model, x, y, nb_classes, eps,
                                clip_min, clip_max, eps_iter, nb_iter,
                                report_path,
                                batch_size=BATCH_SIZE,
                                eps_iter_small=None):
  """A reasonable attack bundling recipe for a max norm threat model and
  a defender that uses confidence thresholding.

  References:
  https://openreview.net/forum?id=H1g0piA9tQ

  This version runs each attack a fixed number of times.
  It is more exhaustive than `single_run_max_confidence_recipe` but because
  it uses a fixed budget rather than running indefinitely it is more
  appropriate for making fair comparisons between two models.

  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param nb_classes: int, number of classes
  :param eps: float, maximum size of perturbation (measured by max norm)
  :param eps_iter: float, step size for one version of PGD attacks
    (will also run another version with smaller step size)
  :param nb_iter: int, number of iterations for one version of PGD attacks
    (will also run another version with 25X more iterations)
  :param report_path: str, the path that the report will be saved to.
  :batch_size: int, the total number of examples to run simultaneously
  :param eps_iter_small: float, the step size to use for more expensive version of the attack.
    If not specified, usess eps_iter / 25
  """
  noise_attack = Noise(model, sess)
  pgd_attack = ProjectedGradientDescent(model, sess)
  threat_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max}
  noise_attack_config = AttackConfig(noise_attack, threat_params)
  attack_configs = [noise_attack_config]
  pgd_attack_configs = []
  pgd_params = copy.copy(threat_params)
  pgd_params["eps_iter"] = eps_iter
  pgd_params["nb_iter"] = nb_iter
  assert batch_size % num_devices == 0
  dev_batch_size = batch_size // num_devices
  ones = tf.ones(dev_batch_size, tf.int32)
  if eps_iter_small is None:
    eps_iter_small = eps_iter / 25.
  expensive_pgd = []
  for cls in range(nb_classes):
    cls_params = copy.copy(pgd_params)
    cls_params['y_target'] = tf.to_float(tf.one_hot(ones * cls, nb_classes))
    cls_attack_config = AttackConfig(pgd_attack, cls_params, "pgd_" + str(cls))
    pgd_attack_configs.append(cls_attack_config)
    expensive_params = copy.copy(cls_params)
    expensive_params["eps_iter"] = eps_iter_small
    expensive_params["nb_iter"] *= 25.
    expensive_config = AttackConfig(
        pgd_attack, expensive_params, "expensive_pgd_" + str(cls))
    expensive_pgd.append(expensive_config)
  attack_configs = [noise_attack_config] + pgd_attack_configs + expensive_pgd
  new_work_goal = {config: 5 for config in attack_configs}
  pgd_work_goal = {config: 5 for config in pgd_attack_configs}
  # TODO: lower priority: make sure bundler won't waste time running targeted
  # attacks on examples where the target class is the true class.
  goals = [Misclassify(new_work_goal={noise_attack_config: 50}),
           Misclassify(new_work_goal=pgd_work_goal),
           MaxConfidence(t=0.5, new_work_goal=new_work_goal),
           MaxConfidence(t=0.75, new_work_goal=new_work_goal),
           MaxConfidence(t=0.875, new_work_goal=new_work_goal),
           MaxConfidence(t=0.9375, new_work_goal=new_work_goal),
           MaxConfidence(t=0.96875, new_work_goal=new_work_goal),
           MaxConfidence(t=0.984375, new_work_goal=new_work_goal),
           MaxConfidence(t=1., new_work_goal=new_work_goal)]
  bundle_attacks(sess, model, x, y, attack_configs, goals, report_path)


def random_search_max_confidence_recipe(sess, model, x, y, eps,
                                        clip_min, clip_max,
                                        report_path, batch_size=BATCH_SIZE,
                                        num_noise_points=10000):
  """Max confidence using random search.

  References:
  https://openreview.net/forum?id=H1g0piA9tQ
    Describes the max_confidence procedure used for the bundling in this recipe
  https://arxiv.org/abs/1802.00420
    Describes using random search with 1e5 or more random points to avoid
    gradient masking.

  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param nb_classes: int, number of classes
  :param eps: float, maximum size of perturbation (measured by max norm)
  :param eps_iter: float, step size for one version of PGD attacks
    (will also run another version with 25X smaller step size)
  :param nb_iter: int, number of iterations for one version of PGD attacks
    (will also run another version with 25X more iterations)
  :param report_path: str, the path that the report will be saved to.
  :batch_size: int, the total number of examples to run simultaneously
  """
  noise_attack = Noise(model, sess)
  threat_params = {"eps": eps, "clip_min": clip_min, "clip_max": clip_max}
  noise_attack_config = AttackConfig(noise_attack, threat_params)
  attack_configs = [noise_attack_config]
  assert batch_size % num_devices == 0
  new_work_goal = {noise_attack_config: num_noise_points}
  goals = [MaxConfidence(t=1., new_work_goal=new_work_goal)]
  bundle_attacks(sess, model, x, y, attack_configs, goals, report_path)


class AttackConfig(object):
  """
  An attack and associated parameters.
  :param attack: cleverhans.attacks.Attack
  :param params: dict of keyword arguments to pass to attack.generate
  :param name: str, name to be returned by __str__ / __repr__
  :param pass_y: bool, whether to pass y to `attack.generate`
  """

  def __init__(self, attack, params=None, name=None, pass_y=False):
    self.attack = attack
    self.params = params
    self.name = name
    if params is not None:
      assert isinstance(params, dict)
      for key in params:
        assert isinstance(key, six.string_types), type(key)
    self.pass_y = pass_y

  def __str__(self):
    if self.name is not None:
      return self.name
    return "AttackConfig(" + str(self.attack) + ", " + str(self.params) + ")"

  def __repr__(self):
    return self.__str__()


def bundle_attacks(sess, model, x, y, attack_configs, goals, report_path,
                   attack_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE):
  """
  Runs attack bundling.
  Users of cleverhans may call this function but are more likely to call
  one of the recipes above.

  Reference: https://openreview.net/forum?id=H1g0piA9tQ

  :param sess: tf.session.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param attack_configs: list of AttackConfigs to run
  :param goals: list of AttackGoals to run
    The bundler works through the goals in order, until each is satisfied.
    Some goals may never be satisfied, in which case the bundler will run
    forever, updating the report on disk as it goes.
  :param report_path: str, the path the report will be saved to
  :param attack_batch_size: int, batch size for generating adversarial examples
  :param eval_batch_size: int, batch size for evaluating the model on clean / adversarial examples
  :returns:
    adv_x: The adversarial examples, in the same format as `x`
    run_counts: dict mapping each AttackConfig to a numpy array reporting
      how many times that AttackConfig was run on each example
  """
  assert isinstance(sess, tf.Session)
  assert isinstance(model, Model)
  assert all(isinstance(attack_config, AttackConfig) for attack_config
             in attack_configs)
  assert all(isinstance(goal, AttackGoal) for goal in goals)
  assert isinstance(report_path, six.string_types)
  if x.shape[0] != y.shape[0]:
    raise ValueError("Number of input examples does not match number of labels")

  # Note: no need to precompile attacks, correctness_and_confidence
  # caches them

  run_counts = {}
  for attack_config in attack_configs:
    run_counts[attack_config] = np.zeros(x.shape[0], dtype=np.int64)

  # TODO: make an interface to pass this in if it has already been computed
  # elsewhere
  _logger.info("Running on clean data to initialize the report...")
  packed = correctness_and_confidence(sess, model, x, y, batch_size=eval_batch_size,
                                      devices=devices)
  _logger.info("...done")
  correctness, confidence = packed
  _logger.info("Accuracy: " + str(correctness.mean()))
  report = ConfidenceReport()
  report['clean'] = ConfidenceReportEntry(correctness, confidence)

  adv_x = x.copy()

  for goal in goals:
    bundle_attacks_with_goal(sess, model, x, y, adv_x, attack_configs,
                             run_counts,
                             goal, report, report_path,
                             attack_batch_size=attack_batch_size, eval_batch_size=eval_batch_size)

  # Many users will set `goals` to make this run forever, so the return
  # statement is not the primary way to get information out.
  return adv_x, run_counts

def bundle_attacks_with_goal(sess, model, x, y, adv_x, attack_configs,
                             run_counts,
                             goal, report, report_path,
                             attack_batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE):
  """
  Runs attack bundling, working on one specific AttackGoal.
  This function is mostly intended to be called by `bundle_attacks`.

  Reference: https://openreview.net/forum?id=H1g0piA9tQ

  :param sess: tf.session.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param adv_x: numpy array containing the adversarial examples made so far
    by earlier work in the bundling process
  :param attack_configs: list of AttackConfigs to run
  :param run_counts: dict mapping AttackConfigs to numpy arrays specifying
    how many times they have been run on each example
  :param goal: AttackGoal to run
  :param report: ConfidenceReport
  :param report_path: str, the path the report will be saved to
  :param attack_batch_size: int, batch size for generating adversarial examples
  :param eval_batch_size: int, batch size for evaluating the model on adversarial examples
  """
  goal.start(run_counts)
  _logger.info("Running criteria for new goal...")
  criteria = goal.get_criteria(sess, model, adv_x, y, batch_size=eval_batch_size)
  assert 'correctness' in criteria
  _logger.info("Accuracy: " + str(criteria['correctness'].mean()))
  assert 'confidence' in criteria
  while not goal.is_satisfied(criteria, run_counts):
    run_batch_with_goal(sess, model, x, y, adv_x, criteria, attack_configs,
                        run_counts,
                        goal, report, report_path,
                        attack_batch_size=attack_batch_size)
  # Save after finishing all goals.
  # The incremental saves run on a timer. This save is needed so that the last
  # few attacks after the timer don't get discarded
  report.completed = True
  save(criteria, report, report_path, adv_x)


def run_batch_with_goal(sess, model, x, y, adv_x_val, criteria, attack_configs,
                        run_counts, goal, report, report_path,
                        attack_batch_size=BATCH_SIZE):
  """
  Runs attack bundling on one batch of data.
  This function is mostly intended to be called by
  `bundle_attacks_with_goal`.

  :param sess: tf.session.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param adv_x_val: numpy array containing the adversarial examples made so far
    by earlier work in the bundling process
  :param criteria: dict mapping string names of criteria to numpy arrays with
    their values for each example
    (Different AttackGoals track different criteria)
  :param run_counts: dict mapping AttackConfigs to numpy arrays reporting how
    many times they have been run on each example
  :param goal: the AttackGoal to work on
  :param report: dict, see `bundle_attacks_with_goal`
  :param report_path: str, path to save the report to
  """
  attack_config = goal.get_attack_config(attack_configs, run_counts, criteria)
  idxs = goal.request_examples(attack_config, criteria, run_counts,
                               attack_batch_size)
  x_batch = x[idxs]
  assert x_batch.shape[0] == attack_batch_size
  y_batch = y[idxs]
  assert y_batch.shape[0] == attack_batch_size
  adv_x_batch = run_attack(sess, model, x_batch, y_batch,
                           attack_config.attack, attack_config.params,
                           attack_batch_size, devices, pass_y=attack_config.pass_y)
  criteria_batch = goal.get_criteria(sess, model, adv_x_batch, y_batch,
                                     batch_size=min(attack_batch_size,
                                                    BATCH_SIZE))
  # This can't be parallelized because some orig examples are copied more
  # than once into the batch
  cur_run_counts = run_counts[attack_config]
  for batch_idx, orig_idx in enumerate(idxs):
    cur_run_counts[orig_idx] += 1
    should_copy = goal.new_wins(criteria, orig_idx, criteria_batch, batch_idx)
    if should_copy:
      adv_x_val[orig_idx] = adv_x_batch[batch_idx]
      for key in criteria:
        criteria[key][orig_idx] = criteria_batch[key][batch_idx]
      assert np.allclose(y[orig_idx], y_batch[batch_idx])
  report['bundled'] = ConfidenceReportEntry(criteria['correctness'], criteria['confidence'])

  should_save = False
  new_time = time.time()
  if hasattr(report, 'time'):
    if new_time - report.time > REPORT_TIME_INTERVAL:
      should_save = True
  else:
    should_save = True
  if should_save:
    report.time = new_time
    goal.print_progress(criteria, run_counts)
    save(criteria, report, report_path, adv_x_val)


def save(criteria, report, report_path, adv_x_val):
  """
  Saves the report and adversarial examples.
  :param criteria: dict, of the form returned by AttackGoal.get_criteria
  :param report: dict containing a confidence report
  :param report_path: string, filepath
  :param adv_x_val: numpy array containing dataset of adversarial examples
  """
  print_stats(criteria['correctness'], criteria['confidence'], 'bundled')

  print("Saving to " + report_path)
  serial.save(report_path, report)

  assert report_path.endswith(".joblib")
  adv_x_path = report_path[:-len(".joblib")] + "_adv.npy"
  np.save(adv_x_path, adv_x_val)


class AttackGoal(object):
  """Specifies goals for attack bundling.
  Different bundling recipes can have different priorities.
  - When choosing which examples to attack in the next batch, do we want
  to focus on examples that are not yet misclassified? Among the
  still correctly classified ones, do we want to focus on the ones that
  have not been attacked many times yet? Do we want to focus on the ones
  that have low loss / low confidence so far?
  - After an attack has been run, do we prefer the new adversarial example
  or the old one? Is the new one better if it causes higher confidence
  in the wrong prediction? If it succeeds with a smaller perturbation?
  For different use cases, the answers to these questions is different.
  Implement different AttackGoal subclasses to represent the priorities
  for your use case.
  """

  def start(self, run_counts):
    """
    Called by the bundler when it starts working on the goal.

    :param run_counts: dict mapping AttackConfigs to numpy arrays reporting
      how many times they have been run on each example.
    """

  def get_criteria(self, sess, model, advx, y, batch_size=BATCH_SIZE):
    """
    Returns a dictionary mapping the name of each criterion to a NumPy
    array containing the value of that criterion for each adversarial
    example.
    Subclasses can add extra criteria by implementing the `extra_criteria`
    method.

    :param sess: tf.session.Session
    :param model: cleverhans.model.Model
    :param adv_x: numpy array containing the adversarial examples made so far
      by earlier work in the bundling process
    :param y: numpy array containing true labels
    :param batch_size: int, batch size
    """

    names, factory = self.extra_criteria()
    factory = _CriteriaFactory(model, factory)
    results = batch_eval_multi_worker(sess, factory, [advx, y],
                                      batch_size=batch_size, devices=devices)
    names = ['correctness', 'confidence'] + names
    out = dict(safe_zip(names, results))
    return out

  def extra_criteria(self):
    """
    Subclasses implement this to specify any extra criteria they need to track.
    : returns: list of criterion names, _ExtraCriteriaFactory implementing them
    """
    return [], None

  def request_examples(self, attack_config, criteria, run_counts, batch_size):
    """
    Returns a numpy array of integer example indices to run in the next batch.
    """
    raise NotImplementedError(str(type(self)) +
                              "needs to implement request_examples")

  def is_satisfied(self, criteria, run_counts):
    """
    Returns a bool indicating whether the goal has been satisfied.
    """
    raise NotImplementedError(str(type(self)) +
                              " needs to implement is_satisfied.")

  def print_progress(self, criteria, run_counts):
    """
    Prints a progress message about how much has been done toward the goal.
    :param criteria: dict, of the format returned by get_criteria
    :param run_counts: dict mapping each AttackConfig to a numpy array
      specifying how many times it has been run for each example
    """
    print("Working on a " + self.__class__.__name__ + " goal.")

  def get_attack_config(self, attack_configs, run_counts, criteria):
    """
    Returns an AttackConfig to run on the next batch.
    """
    raise NotImplementedError(str(type(self)) +
                              " needs to implement get_attack_config")

  def new_wins(self, orig_criteria, orig_idx, new_criteria, new_idx):
    """
    Returns a bool indicating whether a new adversarial example is better
    than the pre-existing one for the same clean example.
    :param orig_criteria: dict mapping names of criteria to their value
      for each example in the whole dataset
    :param orig_idx: The position of the pre-existing example within the
      whole dataset.
    :param new_criteria: dict, like orig_criteria, but with values only
      on the latest batch of adversarial examples
    :param new_idx: The position of the new adversarial example within
      the batch
    """
    raise NotImplementedError(str(type(self))
                              + " needs to implement new_wins.")


class Misclassify(AttackGoal):
  """An AttackGoal that prioritizes misclassifying all examples.

  Times out when each attack has been run the requested number of times.
  Some examples may be attacked more than the goal number because we
  always run a full batch of attacks and sometimes the batch size is
  larger than the number of examples with attacks left to do.
  :param new_work_goal: dict
    Maps AttackConfigs to ints specifying how many times they should be
    run before timing out.
    If this dict is not specified, all attacks will be run, repeatedly,
    until all examples are misclassified (or forever if some cannot
    be changed into misclassified examples).
    If this dict is specfied, only attacks in the dict will be run.
  :param break_ties: string name of tie-breaking scheme for `new_wins`
    When two examples are misclassified, how do we choose between them?
    Currently the only scheme is 'wrong_confidence', where we prefer the
    one with higher confidence assigned to a single wrong class.
    In the future we may support other schemes like smaller perturbation
    size, higher loss, etc.
  """

  def __init__(self, new_work_goal=None, break_ties='wrong_confidence'):
    super(Misclassify, self).__init__()
    self.new_work_goal = new_work_goal
    assert all(isinstance(key, AttackConfig) for key in new_work_goal.keys())
    assert all(isinstance(value, int) for value in new_work_goal.values())
    self.rng = np.random.RandomState([2018, 10, 5, 9])
    self.break_ties = break_ties

  def start(self, run_counts):
    for key in run_counts:
      value = run_counts[key]
      assert value.ndim == 1
    _logger.info("Started working on a Misclassify goal")
    self.work_before = deep_copy(run_counts)

  def is_satisfied(self, criteria, run_counts):
    correctness = criteria['correctness']
    assert correctness.dtype == np.bool
    assert correctness.ndim == 1
    if correctness.max() == 0:
      _logger.info("Everything is misclassified! Done with Misclassify goal")
      return True
    if self.new_work_goal is None:
      return False
    correct_run_counts = self.filter(run_counts, criteria)
    correct_work_before = self.filter(self.work_before, criteria)
    unfinished = unfinished_attack_configs(self.new_work_goal,
                                           correct_work_before,
                                           correct_run_counts)
    finished = len(unfinished) == 0
    if finished:
      _logger.info("Misclassify timed out after running all requested attacks")
    else:
      pass
      # _logger.info("Miclassify goal still has attacks to run")
    return finished

  def print_progress(self, criteria, run_counts):
    print("Working on a " + self.__class__.__name__ + " goal.")
    num_below = criteria['correctness'].sum()
    print(str(num_below) + " examples are still correctly classified.")
    if self.new_work_goal is None:
      print("No work goal: running all attacks indefinitely")
    else:
      print("Working until all attacks have been run enough times")
      filtered_run_counts = self.filter(run_counts, criteria)
      filtered_work_before = self.filter(self.work_before, criteria)
      for ac in self.new_work_goal:
        goal = self.new_work_goal[ac]
        new = filtered_run_counts[ac] - filtered_work_before[ac]
        if new.size > 0:
          min_new = new.min()
          if min_new < goal:
            num_min = (new == min_new).sum()
            print("\t" + str(ac) + ": goal of " + str(goal) + " runs, but "
                  + str(num_min) + " examples have been run only " + str(min_new)
                  + " times")

  def filter(self, run_counts, criteria):
    """
    Return run counts only for examples that are still correctly classified
    """
    correctness = criteria['correctness']
    assert correctness.dtype == np.bool
    filtered_counts = deep_copy(run_counts)
    for key in filtered_counts:
      filtered_counts[key] = filtered_counts[key][correctness]
    return filtered_counts

  def get_attack_config(self, attack_configs, run_counts, criteria):
    if self.new_work_goal is not None:
      correct_work_before = self.filter(self.work_before, criteria)
      correct_run_counts = self.filter(run_counts, criteria)
      attack_configs = unfinished_attack_configs(self.new_work_goal,
                                                 correct_work_before,
                                                 correct_run_counts)
    attack_config = attack_configs[self.rng.randint(len(attack_configs))]
    return attack_config

  def extra_criteria(self):
    if self.break_ties == "wrong_confidence":
      return ["wrong_confidence"], _WrongConfidenceFactory()
    else:
      raise NotImplementedError()

  def request_examples(self, attack_config, criteria, run_counts, batch_size):
    correctness = criteria['correctness']
    assert correctness.dtype == np.bool
    total = correctness.size
    total_correct = correctness.sum()
    all_idxs = np.arange(total)
    run_counts = run_counts[attack_config]
    if total_correct > 0:
      correct_idxs = all_idxs[correctness]
      assert correct_idxs.size == total_correct
      run_counts = run_counts[correctness]
      pairs = safe_zip(correct_idxs, run_counts)
    else:
      pairs = safe_zip(all_idxs, run_counts)
    # In PY3, pairs is now an iterator.
    # To support sorting, we need to make it a list.
    pairs = list(pairs)

    def key(pair):
      return pair[1]
    pairs.sort(key=key)
    idxs = [pair[0] for pair in pairs]
    while len(idxs) < batch_size:
      needed = batch_size - len(idxs)
      idxs = idxs + idxs[:needed]
    if len(idxs) > batch_size:
      idxs = idxs[:batch_size]
    idxs = np.array(idxs)
    return idxs

  def new_wins(self, orig_criteria, orig_idx, new_criteria, new_idx):
    orig_correct = orig_criteria['correctness'][orig_idx]
    new_correct = new_criteria['correctness'][new_idx]
    if orig_correct and not new_correct:
      return True
    if (not orig_correct) and new_correct:
      return False
    assert orig_correct == new_correct
    if self.break_ties == "wrong_confidence":
      new = new_criteria["wrong_confidence"][new_idx]
      orig = orig_criteria['wrong_confidence'][orig_idx]
      return new > orig
    else:
      raise NotImplementedError(self.break_ties)


class MaxConfidence(AttackGoal):
  """
  The AttackGoal corresponding the MaxConfidence procedure.

  Reference: https://openreview.net/forum?id=H1g0piA9tQ

  This should be used with a recipe that includes AttackConfigs
  that target all of the classes, plus an any additional AttackConfigs
  that may help to avoid gradient masking.

  This AttackGoal prioritizes getting all examples above a specified
  threshold. (If the threshold is set to 1, then no examples are above
  the threshold, so all are attacked equally often). The MaxConfidence
  attack procedure against *a single example* is optimal regardless of
  threshold, so long as t >= 0.5, but when attacking a population of
  examples with finite computation time, knowledge of the threshold is
  necessary to determine which examples to prioritize attacking.

  :param t: Prioritize pushing examples above this threshold.
  :param new_work_goal: Optional dict mapping AttackConfigs to ints.
    The int specifies the number of times to run each AttackConfig on each
    below-threshold example before giving up.
    If not specified, this goal runs all available attacks and never gives
    up.
  """

  def __init__(self, t=1., new_work_goal=None):
    super(MaxConfidence, self).__init__()
    self.t = t
    self.new_work_goal = new_work_goal
    if new_work_goal is not None:
      for key in new_work_goal:
        assert isinstance(key, AttackConfig)
        assert isinstance(new_work_goal[key], int)
    self.rng = np.random.RandomState([2018, 10, 7, 12])

  def filter(self, run_counts, criteria):
    """
    Return the counts for only those examples that are below the threshold
    """
    wrong_confidence = criteria['wrong_confidence']
    below_t = wrong_confidence <= self.t
    filtered_counts = deep_copy(run_counts)
    for key in filtered_counts:
      filtered_counts[key] = filtered_counts[key][below_t]
    return filtered_counts

  def extra_criteria(self):
    return ["wrong_confidence"], _WrongConfidenceFactory()

  def is_satisfied(self, criteria, run_counts):
    wrong_confidence = criteria['wrong_confidence']
    if wrong_confidence.min() > self.t:
      _logger.info("Everything is above threshold " + str(self.t))
      _logger.info("Done with MaxConfidence goal")
      return True
    if self.new_work_goal is None:
      return False
    filtered_run_counts = self.filter(run_counts, criteria)
    filtered_work_before = self.filter(self.work_before, criteria)
    unfinished = unfinished_attack_configs(self.new_work_goal,
                                           filtered_work_before,
                                           filtered_run_counts,
                                           log=False)
    finished = len(unfinished) == 0
    if finished:
      _logger.info(
          "MaxConfidence timed out after running all requested attacks")
    else:
      pass
    return finished

  def print_progress(self, criteria, run_counts):
    print("Working on a " + self.__class__.__name__ + " goal.")
    if self.t == 1.:
      print("Threshold of 1, so just driving up confidence of all examples.")
    else:
      print("Target threshold of " + str(self.t))
      num_below = (criteria['wrong_confidence'] <= self.t).sum()
      print(str(num_below) + " examples are below the target threshold.")
    if self.new_work_goal is None:
      print("No work goal: running all attacks indefinitely")
    else:
      print("Working until all attacks have been run enough times")
      filtered_run_counts = self.filter(run_counts, criteria)
      filtered_work_before = self.filter(self.work_before, criteria)
      for ac in self.new_work_goal:
        goal = self.new_work_goal[ac]
        new = filtered_run_counts[ac] - filtered_work_before[ac]
        min_new = new.min()
        if min_new < goal:
          num_min = (new == min_new).sum()
          print("\t" + str(ac) + ": goal of " + str(goal) + " runs, but "
                + str(num_min) + " examples have been run only " + str(min_new)
                + " times")

  def get_attack_config(self, attack_configs, run_counts, criteria):
    # TODO: refactor to avoid this duplicated method
    if self.new_work_goal is not None:
      correct_work_before = self.filter(self.work_before, criteria)
      correct_run_counts = self.filter(run_counts, criteria)
      attack_configs = unfinished_attack_configs(self.new_work_goal,
                                                 correct_work_before,
                                                 correct_run_counts)
    attack_config = attack_configs[self.rng.randint(len(attack_configs))]
    return attack_config

  def start(self, run_counts):
    _logger.info("Started working on a MaxConfidence goal")
    _logger.info("Threshold: " + str(self.t))
    if self.new_work_goal is None:
      if self.t >= 1.:
        _logger.info("This goal will run forever")
      else:
        _logger.info("This goal will run until all examples have confidence"
                     + " greater than " + str(self.t) + ", which may never"
                     + " happen.")
    self.work_before = deep_copy(run_counts)

  def request_examples(self, attack_config, criteria, run_counts, batch_size):
    wrong_confidence = criteria['wrong_confidence']
    below_t = wrong_confidence <= self.t
    assert below_t.dtype == np.bool
    total = below_t.size
    total_below = below_t.sum()
    all_idxs = np.arange(total)
    run_counts = run_counts[attack_config]
    if total_below > 0:
      correct_idxs = all_idxs[below_t]
      assert correct_idxs.size == total_below
      run_counts = run_counts[below_t]
      pairs = safe_zip(correct_idxs, run_counts)
    else:
      pairs = safe_zip(all_idxs, run_counts)

    def key(pair):
      return pair[1]
    pairs.sort(key=key)
    idxs = [pair[0] for pair in pairs]
    while len(idxs) < batch_size:
      needed = batch_size - len(idxs)
      idxs = idxs + idxs[:needed]
    if len(idxs) > batch_size:
      idxs = idxs[:batch_size]
    idxs = np.array(idxs)
    return idxs

  def new_wins(self, orig_criteria, orig_idx, new_criteria, new_idx):
    new_wrong_confidence = new_criteria['wrong_confidence'][new_idx]
    orig_wrong_confidence = orig_criteria['wrong_confidence'][orig_idx]
    return new_wrong_confidence > orig_wrong_confidence


def unfinished_attack_configs(new_work_goal, work_before, run_counts,
                              log=False):
  """
  Returns a list of attack configs that have not yet been run the desired
  number of times.
  :param new_work_goal: dict mapping attacks to desired number of times to run
  :param work_before: dict mapping attacks to number of times they were run
    before starting this new goal. Should be prefiltered to include only
    examples that don't already meet the primary goal
  :param run_counts: dict mapping attacks to total number of times they have
    ever been run. Should be prefiltered to include only examples that don't
    already meet the primary goal
  """

  assert isinstance(work_before, dict), work_before

  for key in work_before:
    value = work_before[key]
    assert value.ndim == 1, value.shape
    if key in run_counts:
      assert run_counts[key].shape == value.shape

  attack_configs = []
  for attack_config in new_work_goal:
    done_now = run_counts[attack_config]
    if log:
      _logger.info(str(attack_config) +
                   " ave run count: " + str(done_now.mean()))
      _logger.info(str(attack_config) +
                   " min run count: " + str(done_now.min()))
    done_before = work_before[attack_config]
    if log:
      _logger.info(str(attack_config) + " mean work before: " +
                   str(done_before.mean()))
    # This is the vector for all examples
    new = done_now - done_before
    # The work is only done when it has been done for every example
    new = new.min()
    assert isinstance(new, (int, np.int64)), type(new)
    new_goal = new_work_goal[attack_config]
    assert isinstance(new_goal, int), type(new_goal)
    if new < new_goal:
      if log:
        _logger.info(str(attack_config) + " has run " +
                     str(new) + " of " + str(new_goal))
      attack_configs.append(attack_config)
  return attack_configs


class _CriteriaFactory(object):
  """
  A factory that builds the expression to evaluate all criteria.
  """

  def __init__(self, model, extra_criteria_factory=None):
    self.model = model
    self.extra_criteria_factory = extra_criteria_factory
    properties_to_hash = (model, )
    if extra_criteria_factory is not None:
      if extra_criteria_factory.properties_to_hash is not None:
        extra_properties = extra_criteria_factory.properties_to_hash
        properties_to_hash = properties_to_hash + extra_properties
    self.properties_to_hash = properties_to_hash

  def __hash__(self):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    return self.properties_to_hash.__hash__()

  def __eq__(self, other):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    if not isinstance(other, _CriteriaFactory):
      return False
    if (type(self.extra_criteria_factory) is not
        type(other.extra_criteria_factory)):
      return False
    return self.properties_to_hash == other.properties_to_hash

  def __call__(self):
    x_batch = self.model.make_input_placeholder()
    y_batch = self.model.make_label_placeholder()

    predictions = self.model.get_probs(x_batch)
    correct = tf.equal(tf.argmax(y_batch, axis=-1),
                       tf.argmax(predictions, axis=-1))
    max_probs = tf.reduce_max(predictions, axis=1)

    if self.extra_criteria_factory is not None:
      extra_criteria = self.extra_criteria_factory(x_batch, y_batch,
                                                   predictions, correct,
                                                   max_probs)
    else:
      extra_criteria = tuple([])

    return (x_batch, y_batch), (correct, max_probs) + extra_criteria


class _ExtraCriteriaFactory(object):
  """
  A factory that builds extra criteria
  """

  def __init__(self, properties_to_hash=None):
    self.properties_to_hash = properties_to_hash

  def __hash__(self):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    return self.properties_to_hash.__hash__()

  def __eq__(self, other):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    if not isinstance(other, _ExtraCriteriaFactory):
      return False
    return self.properties_to_hash == other.properties_to_hash

  def __call__(self, x_batch, y_batch, predictions, correct, max_probs):
    raise NotImplementedError()


class _WrongConfidenceFactory(_ExtraCriteriaFactory):
  def __call__(self, x_batch, y_batch, predictions, correct, max_probs):
    max_wrong_probs = tf.reduce_max(predictions * (1. - y_batch), axis=1)
    return tuple([max_wrong_probs])


def bundle_examples_with_goal(sess, model, adv_x_list, y, goal,
                              report_path, batch_size=BATCH_SIZE):
  """
  A post-processor version of attack bundling, that chooses the strongest
  example from the output of multiple earlier bundling strategies.

  :param sess: tf.session.Session
  :param model: cleverhans.model.Model
  :param adv_x_list: list of numpy arrays
    Each entry in the list is the output of a previous bundler; it is an
      adversarial version of the whole dataset.
  :param y: numpy array containing true labels
  :param goal: AttackGoal to use to choose the best version of each adversarial
    example
  :param report_path: str, the path the report will be saved to
  :param batch_size: int, batch size
  """

  # Check the input
  num_attacks = len(adv_x_list)
  assert num_attacks > 0
  adv_x_0 = adv_x_list[0]
  assert isinstance(adv_x_0, np.ndarray)
  assert all(adv_x.shape == adv_x_0.shape for adv_x in adv_x_list)

  # Allocate the output
  out = np.zeros_like(adv_x_0)
  m = adv_x_0.shape[0]
  # Initialize with negative sentinel values to make sure everything is
  # written to
  correctness = -np.ones(m, dtype='int32')
  confidence = -np.ones(m, dtype='float32')

  # Gather criteria
  criteria = [goal.get_criteria(sess, model, adv_x, y, batch_size=batch_size) for adv_x in adv_x_list]
  assert all('correctness' in c for c in criteria)
  assert all('confidence' in c for c in criteria)
  _logger.info("Accuracy on each advx dataset: ")
  for c in criteria:
    _logger.info("\t" + str(c['correctness'].mean()))

  for example_idx in range(m):
    # Index of the best attack for this example
    attack_idx = 0
    # Find the winner
    for candidate_idx in range(1, num_attacks):
      if goal.new_wins(criteria[attack_idx], example_idx,
                       criteria[candidate_idx], example_idx):
        attack_idx = candidate_idx
    # Copy the winner into the output
    out[example_idx] = adv_x_list[attack_idx][example_idx]
    correctness[example_idx] = criteria[attack_idx]['correctness'][example_idx]
    confidence[example_idx] = criteria[attack_idx]['confidence'][example_idx]

  assert correctness.min() >= 0
  assert correctness.max() <= 1
  assert confidence.min() >= 0.
  assert confidence.max() <= 1.
  correctness = correctness.astype('bool')
  _logger.info("Accuracy on bundled examples: " + str(correctness.mean()))

  report = ConfidenceReport()
  report['bundled'] = ConfidenceReportEntry(correctness, confidence)
  serial.save(report_path, report)
  assert report_path.endswith('.joblib')
  adv_x_path = report_path[:-len('.joblib')] + "_adv_x.npy"
  np.save(adv_x_path, out)

def spsa_max_confidence_recipe(sess, model, x, y, nb_classes, eps,
                               clip_min, clip_max, nb_iter,
                               report_path,
                               spsa_samples=SPSA.DEFAULT_SPSA_SAMPLES,
                               spsa_iters=SPSA.DEFAULT_SPSA_ITERS,
                               eval_batch_size=BATCH_SIZE):
  """Runs the MaxConfidence attack using SPSA as the underlying optimizer.

  Even though this runs only one attack, it must be implemented as a bundler
  because SPSA supports only batch_size=1. The cleverhans.attacks.MaxConfidence
  attack internally multiplies the batch size by nb_classes, so it can't take
  SPSA as a base attacker. Insteader, we must bundle batch_size=1 calls using
  cleverhans.attack_bundling.MaxConfidence.

  References:
  https://openreview.net/forum?id=H1g0piA9tQ

  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing clean example inputs to attack
  :param y: numpy array containing true labels
  :param nb_classes: int, number of classes
  :param eps: float, maximum size of perturbation (measured by max norm)
  :param nb_iter: int, number of iterations for one version of PGD attacks
    (will also run another version with 25X more iterations)
  :param report_path: str, the path that the report will be saved to.
  :param eval_batch_size: int, batch size for evaluation (as opposed to making attacks)
  """
  spsa = SPSA(model, sess)
  spsa_params = {"eps": eps, "clip_min" : clip_min, "clip_max" : clip_max,
                 "nb_iter": nb_iter, "spsa_samples": spsa_samples,
                 "spsa_iters": spsa_iters}
  attack_configs = []
  dev_batch_size = 1 # The only batch size supported by SPSA
  batch_size = num_devices
  ones = tf.ones(dev_batch_size, tf.int32)
  for cls in range(nb_classes):
    cls_params = copy.copy(spsa_params)
    cls_params['y_target'] = tf.to_float(tf.one_hot(ones * cls, nb_classes))
    cls_attack_config = AttackConfig(spsa, cls_params, "spsa_" + str(cls))
    attack_configs.append(cls_attack_config)
  new_work_goal = {config: 1 for config in attack_configs}
  goals = [MaxConfidence(t=1., new_work_goal=new_work_goal)]
  bundle_attacks(sess, model, x, y, attack_configs, goals, report_path,
                 attack_batch_size=batch_size, eval_batch_size=eval_batch_size)
