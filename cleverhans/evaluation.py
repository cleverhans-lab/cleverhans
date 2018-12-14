"""
Functionality for evaluating expressions across entire datasets.
Includes multi-GPU support for fast evaluation.
"""

from distutils.version import LooseVersion
import warnings
import numpy as np
from six.moves import range
import tensorflow as tf

import cleverhans
from cleverhans import canary
from cleverhans.utils import create_logger
from cleverhans.utils_tf import infer_devices


def accuracy(sess, model, x, y, batch_size=None, devices=None, feed=None,
             attack=None, attack_params=None):
  """
  Compute the accuracy of a TF model on some data
  :param sess: TF session to use when training the graph
  :param model: cleverhans.model.Model instance
  :param x: numpy array containing input examples (e.g. MNIST().x_test )
  :param y: numpy array containing example labels (e.g. MNIST().y_test )
  :param batch_size: Number of examples to use in a single evaluation batch.
      If not specified, this function will use a reasonable guess and
      may run out of memory.
      When choosing the batch size, keep in mind that the batch will
      be divided up evenly among available devices. If you can fit 128
      examples in memory on one GPU and you have 8 GPUs, you probably
      want to use a batch size of 1024 (unless a different batch size
      runs faster with the ops you are using, etc.)
  :param devices: An optional list of string device names to use.
    If not specified, this function will use all visible GPUs.
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param attack: cleverhans.attack.Attack
    Optional. If no attack specified, evaluates the model on clean data.
    If attack is specified, evaluates the model on adversarial examples
    created by the attack.
  :param attack_params: dictionary
    If attack is specified, this dictionary is passed to attack.generate
    as keyword arguments.
  :return: a float with the accuracy value
  """

  _check_x(x)
  _check_y(y)
  if x.shape[0] != y.shape[0]:
    raise ValueError("Number of input examples and labels do not match.")

  factory = _CorrectFactory(model, attack, attack_params)

  correct, = batch_eval_multi_worker(sess, factory, [x, y],
                                     batch_size=batch_size, devices=devices,
                                     feed=feed)

  return correct.mean()


def class_and_confidence(sess, model, x, y=None, batch_size=None,
                         devices=None, feed=None, attack=None,
                         attack_params=None):
  """
  Return the model's classification of the input data, and the confidence
  (probability) assigned to each example.
  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing input examples (e.g. MNIST().x_test )
  :param y: numpy array containing true labels
    (Needed only if using an attack that avoids these labels)
  :param batch_size: Number of examples to use in a single evaluation batch.
      If not specified, this function will use a reasonable guess and
      may run out of memory.
      When choosing the batch size, keep in mind that the batch will
      be divided up evenly among available devices. If you can fit 128
      examples in memory on one GPU and you have 8 GPUs, you probably
      want to use a batch size of 1024 (unless a different batch size
      runs faster with the ops you are using, etc.)
  :param devices: An optional list of string device names to use.
    If not specified, this function will use all visible GPUs.
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param attack: cleverhans.attack.Attack
    Optional. If no attack specified, evaluates the model on clean data.
    If attack is specified, evaluates the model on adversarial examples
    created by the attack.
  :param attack_params: dictionary
    If attack is specified, this dictionary is passed to attack.generate
    as keyword arguments.
  :return:
    an ndarray of ints indicating the class assigned to each example
    an ndarray of probabilities assigned to the prediction for each example
  """

  _check_x(x)
  inputs = [x]
  if attack is not None:
    inputs.append(y)
    _check_y(y)
    if x.shape[0] != y.shape[0]:
      raise ValueError("Number of input examples and labels do not match.")

  factory = _ClassAndProbFactory(model, attack, attack_params)

  out = batch_eval_multi_worker(sess, factory, inputs, batch_size=batch_size,
                                devices=devices, feed=feed)

  classes, confidence = out

  assert classes.shape == (x.shape[0],)
  assert confidence.shape == (x.shape[0],)
  min_confidence = confidence.min()
  if min_confidence < 0.:
    raise ValueError("Model does not return valid probabilities: " +
                     str(min_confidence))
  max_confidence = confidence.max()
  if max_confidence > 1.:
    raise ValueError("Model does not return valid probablities: " +
                     str(max_confidence))
  assert confidence.min() >= 0., confidence.min()

  return out


def correctness_and_confidence(sess, model, x, y, batch_size=None,
                               devices=None, feed=None, attack=None,
                               attack_params=None):
  """
  Report whether the model is correct and its confidence on each example in
  a dataset.
  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing input examples (e.g. MNIST().x_test )
  :param y: numpy array containing example labels (e.g. MNIST().y_test )
  :param batch_size: Number of examples to use in a single evaluation batch.
      If not specified, this function will use a reasonable guess and
      may run out of memory.
      When choosing the batch size, keep in mind that the batch will
      be divided up evenly among available devices. If you can fit 128
      examples in memory on one GPU and you have 8 GPUs, you probably
      want to use a batch size of 1024 (unless a different batch size
      runs faster with the ops you are using, etc.)
  :param devices: An optional list of string device names to use.
    If not specified, this function will use all visible GPUs.
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param attack: cleverhans.attack.Attack
    Optional. If no attack specified, evaluates the model on clean data.
    If attack is specified, evaluates the model on adversarial examples
    created by the attack.
  :param attack_params: dictionary
    If attack is specified, this dictionary is passed to attack.generate
    as keyword arguments.
  :return:
    an ndarray of bools indicating whether each example is correct
    an ndarray of probabilities assigned to the prediction for each example
  """

  _check_x(x)
  _check_y(y)
  if x.shape[0] != y.shape[0]:
    raise ValueError("Number of input examples and labels do not match.")

  factory = _CorrectAndProbFactory(model, attack, attack_params)

  out = batch_eval_multi_worker(sess, factory, [x, y], batch_size=batch_size,
                                devices=devices, feed=feed)

  correctness, confidence = out

  assert correctness.shape == (x.shape[0],)
  assert confidence.shape == (x.shape[0],)
  min_confidence = confidence.min()
  if min_confidence < 0.:
    raise ValueError("Model does not return valid probabilities: " +
                     str(min_confidence))
  max_confidence = confidence.max()
  if max_confidence > 1.:
    raise ValueError("Model does not return valid probablities: " +
                     str(max_confidence))
  assert confidence.min() >= 0., confidence.min()

  return out


def run_attack(sess, model, x, y, attack, attack_params, batch_size=None,
               devices=None, feed=None, pass_y=False):
  """
  Run attack on every example in a dataset.
  :param sess: tf.Session
  :param model: cleverhans.model.Model
  :param x: numpy array containing input examples (e.g. MNIST().x_test )
  :param y: numpy array containing example labels (e.g. MNIST().y_test )
  :param attack: cleverhans.attack.Attack
  :param attack_params: dictionary
    passed to attack.generate as keyword arguments.
  :param batch_size: Number of examples to use in a single evaluation batch.
      If not specified, this function will use a reasonable guess and
      may run out of memory.
      When choosing the batch size, keep in mind that the batch will
      be divided up evenly among available devices. If you can fit 128
      examples in memory on one GPU and you have 8 GPUs, you probably
      want to use a batch size of 1024 (unless a different batch size
      runs faster with the ops you are using, etc.)
  :param devices: An optional list of string device names to use.
    If not specified, this function will use all visible GPUs.
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param pass_y: bool. If true pass 'y' to `attack.generate`
  :return:
    an ndarray of bools indicating whether each example is correct
    an ndarray of probabilities assigned to the prediction for each example
  """

  _check_x(x)
  _check_y(y)

  factory = _AttackFactory(model, attack, attack_params, pass_y)

  out, = batch_eval_multi_worker(sess, factory, [x, y], batch_size=batch_size,
                                 devices=devices, feed=feed)
  return out


def batch_eval_multi_worker(sess, graph_factory, numpy_inputs, batch_size=None,
                            devices=None, feed=None):
  """
  Generic computation engine for evaluating an expression across a whole
  dataset, divided into batches.

  This function assumes that the work can be parallelized with one worker
  device handling one batch of data. If you need multiple devices per
  batch, use `batch_eval`.

  The tensorflow graph for multiple workers is large, so the first few
  runs of the graph will be very slow. If you expect to run the graph
  few times (few calls to `batch_eval_multi_worker` that each run few
  batches) the startup cost might dominate the runtime, and it might be
  preferable to use the single worker `batch_eval` just because its
  startup cost will be lower.

  :param sess: tensorflow Session
  :param graph_factory: callable
      When called, returns (tf_inputs, tf_outputs) where:
          tf_inputs is a list of placeholders to feed from the dataset
          tf_outputs is a list of tf tensors to calculate
      Example: tf_inputs is [x, y] placeholders, tf_outputs is [accuracy].
      This factory must make new tensors when called, rather than, e.g.
      handing out a reference to existing tensors.
      This factory must make exactly equivalent expressions every time
      it is called, otherwise the results of `batch_eval` will vary
      depending on how work is distributed to devices.
      This factory must respect "with tf.device()" context managers
      that are active when it is called, otherwise work will not be
      distributed to devices correctly.
  :param numpy_inputs:
      A list of numpy arrays defining the dataset to be evaluated.
      The list should have the same length as tf_inputs.
      Each array should have the same number of examples (shape[0]).
      Example: numpy_inputs is [MNIST().x_test, MNIST().y_test]
  :param batch_size: Number of examples to use in a single evaluation batch.
      If not specified, this function will use a reasonable guess and
      may run out of memory.
      When choosing the batch size, keep in mind that the batch will
      be divided up evenly among available devices. If you can fit 128
      examples in memory on one GPU and you have 8 GPUs, you probably
      want to use a batch size of 1024 (unless a different batch size
      runs faster with the ops you are using, etc.)
  :param devices: List of devices to run on. If unspecified, uses all
      available GPUs if any GPUS are available, otherwise uses CPUs.
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :returns: List of numpy arrays corresponding to the outputs produced by
      the graph_factory
  """
  canary.run_canary()
  global _batch_eval_multi_worker_cache

  devices = infer_devices(devices)

  if batch_size is None:
    # For big models this might result in OOM and then the user
    # should just specify batch_size
    batch_size = len(devices) * DEFAULT_EXAMPLES_PER_DEVICE

  n = len(numpy_inputs)
  assert n > 0
  m = numpy_inputs[0].shape[0]
  for i in range(1, n):
    m_i = numpy_inputs[i].shape[0]
    if m != m_i:
      raise ValueError("All of numpy_inputs must have the same number of examples, but the first one has " + str(m)
                       + " examples and input " + str(i) + " has " + str(m_i) + "examples.")
  out = []

  replicated_tf_inputs = []
  replicated_tf_outputs = []
  p = None

  num_devices = len(devices)
  assert batch_size % num_devices == 0
  device_batch_size = batch_size // num_devices

  cache_key = (graph_factory, tuple(devices))
  if cache_key in _batch_eval_multi_worker_cache:
    # Retrieve graph for multi-GPU inference from cache.
    # This avoids adding tf ops to the graph
    packed = _batch_eval_multi_worker_cache[cache_key]
    replicated_tf_inputs, replicated_tf_outputs = packed
    p = len(replicated_tf_outputs[0])
    assert p > 0
  else:
    # This graph has not been built before.
    # Build it now.

    for device in devices:
      with tf.device(device):
        tf_inputs, tf_outputs = graph_factory()
        assert len(tf_inputs) == n
        if p is None:
          p = len(tf_outputs)
          assert p > 0
        else:
          assert len(tf_outputs) == p
        replicated_tf_inputs.append(tf_inputs)
        replicated_tf_outputs.append(tf_outputs)
    del tf_inputs
    del tf_outputs
    # Store the result in the cache
    packed = replicated_tf_inputs, replicated_tf_outputs
    _batch_eval_multi_worker_cache[cache_key] = packed
  for _ in range(p):
    out.append([])
  flat_tf_outputs = []
  for output in range(p):
    for dev_idx in range(num_devices):
      flat_tf_outputs.append(replicated_tf_outputs[dev_idx][output])

  # pad data to have # examples be multiple of batch size
  # we discard the excess later
  num_batches = int(np.ceil(float(m) / batch_size))
  needed_m = num_batches * batch_size
  excess = needed_m - m
  if excess > m:
    raise NotImplementedError(("Your batch size (%(batch_size)d) is bigger"
                               " than the dataset (%(m)d), this function is "
                               "probably overkill.") % locals())

  def pad(array):
    """Pads an array with replicated examples to have `excess` more entries"""
    if excess > 0:
      array = np.concatenate((array, array[:excess]), axis=0)
    return array
  numpy_inputs = [pad(numpy_input) for numpy_input in numpy_inputs]
  orig_m = m
  m = needed_m

  for start in range(0, m, batch_size):
    batch = start // batch_size
    if batch % 100 == 0 and batch > 0:
      _logger.debug("Batch " + str(batch))

    # Compute batch start and end indices
    end = start + batch_size
    numpy_input_batches = [numpy_input[start:end]
                           for numpy_input in numpy_inputs]
    feed_dict = {}
    for dev_idx, tf_inputs in enumerate(replicated_tf_inputs):
      for tf_input, numpy_input in zip(tf_inputs, numpy_input_batches):
        dev_start = dev_idx * device_batch_size
        dev_end = (dev_idx + 1) * device_batch_size
        value = numpy_input[dev_start:dev_end]
        assert value.shape[0] == device_batch_size
        feed_dict[tf_input] = value
    if feed is not None:
      feed_dict.update(feed)
    flat_output_batches = sess.run(flat_tf_outputs, feed_dict=feed_dict)
    for e in flat_output_batches:
      assert e.shape[0] == device_batch_size, e.shape

    output_batches = []
    for output in range(p):
      o_start = output * num_devices
      o_end = (output + 1) * num_devices
      device_values = flat_output_batches[o_start:o_end]
      assert len(device_values) == num_devices
      output_batches.append(device_values)

    for out_elem, device_values in zip(out, output_batches):
      assert len(device_values) == num_devices, (len(device_values),
                                                 num_devices)
      for device_value in device_values:
        assert device_value.shape[0] == device_batch_size
      out_elem.extend(device_values)

  out = [np.concatenate(x, axis=0) for x in out]
  for e in out:
    assert e.shape[0] == m, e.shape

  # Trim off the examples we used to pad up to batch size
  out = [e[:orig_m] for e in out]
  assert len(out) == p, (len(out), p)

  return out


def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, batch_size=None,
               feed=None,
               args=None):
  """
  A helper function that computes a tensor on numpy inputs by batches.
  This version uses exactly the tensorflow graph constructed by the
  caller, so the caller can place specific ops on specific devices
  to implement model parallelism.
  Most users probably prefer `batch_eval_multi_worker` which maps
  a single-device expression to multiple devices in order to evaluate
  faster by parallelizing across data.

  :param sess: tf Session to use
  :param tf_inputs: list of tf Placeholders to feed from the dataset
  :param tf_outputs: list of tf tensors to calculate
  :param numpy_inputs: list of numpy arrays defining the dataset
  :param batch_size: int, batch size to use for evaluation
      If not specified, this function will try to guess the batch size,
      but might get an out of memory error or run the model with an
      unsupported batch size, etc.
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
              Deprecated and included only for backwards compatibility.
               Should contain `batch_size`
  """

  if args is not None:
    warnings.warn("`args` is deprecated and will be removed on or "
                  "after 2019-03-09. Pass `batch_size` directly.")
    if "batch_size" in args:
      assert batch_size is None
      batch_size = args["batch_size"]

  if batch_size is None:
    batch_size = DEFAULT_EXAMPLES_PER_DEVICE

  n = len(numpy_inputs)
  assert n > 0
  assert n == len(tf_inputs)
  m = numpy_inputs[0].shape[0]
  for i in range(1, n):
    assert numpy_inputs[i].shape[0] == m
  out = []
  for _ in tf_outputs:
    out.append([])
  for start in range(0, m, batch_size):
    batch = start // batch_size
    if batch % 100 == 0 and batch > 0:
      _logger.debug("Batch " + str(batch))

    # Compute batch start and end indices
    start = batch * batch_size
    end = start + batch_size
    numpy_input_batches = [numpy_input[start:end]
                           for numpy_input in numpy_inputs]
    cur_batch_size = numpy_input_batches[0].shape[0]
    assert cur_batch_size <= batch_size
    for e in numpy_input_batches:
      assert e.shape[0] == cur_batch_size

    feed_dict = dict(zip(tf_inputs, numpy_input_batches))
    if feed is not None:
      feed_dict.update(feed)
    numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
    for e in numpy_output_batches:
      assert e.shape[0] == cur_batch_size, e.shape
    for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
      out_elem.append(numpy_output_batch)

  out = [np.concatenate(x, axis=0) for x in out]
  for e in out:
    assert e.shape[0] == m, e.shape
  return out


DEFAULT_EXAMPLES_PER_DEVICE = 128


class _CorrectFactory(object):
  """
  A factory for an expression for one bool per example indicating
  whether each example is correct.
  """

  def __init__(self, model, attack=None, attack_params=None):
    if attack_params is None:
      attack_params = {}
    self.model = model
    self.attack = attack
    self.attack_params = attack_params
    hashable_attack_params = tuple((key, attack_params[key]) for key
                                   in sorted(attack_params.keys()))
    self.properties_to_hash = (model, attack, hashable_attack_params)

  def __hash__(self):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    return self.properties_to_hash.__hash__()

  def __eq__(self, other):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    if not isinstance(other, _CorrectFactory):
      return False
    return self.properties_to_hash == other.properties_to_hash

  def __call__(self):
    x_batch = self.model.make_input_placeholder()
    y_batch = self.model.make_label_placeholder()

    if LooseVersion(tf.__version__) < LooseVersion('1.0.0'):
      raise NotImplementedError()

    if self.attack is None:
      x_input = x_batch
    else:
      attack_params = self.attack_params
      if attack_params is None:
        attack_params = {}
      x_input = self.attack.generate(x_batch, y=y_batch, **attack_params)

    predictions = self.model.get_probs(x_input)
    correct = tf.equal(tf.argmax(y_batch, axis=-1),
                       tf.argmax(predictions, axis=-1))

    return (x_batch, y_batch), (correct,)


class _ClassAndProbFactory(object):
  """
  A factory for an expression for the following tuple per (optionally
  adversarial) example:
    - integer class assigned to the example by the model
    - probability assigned to that prediction
  """

  def __init__(self, model, attack=None, attack_params=None):
    if attack_params is None:
      attack_params = {}
    self.model = model
    self.attack = attack
    self.attack_params = attack_params
    hashable_attack_params = tuple((key, attack_params[key]) for key
                                   in sorted(attack_params.keys()))
    self.properties_to_hash = (model, attack, hashable_attack_params)

  def __hash__(self):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    return self.properties_to_hash.__hash__()

  def __eq__(self, other):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    if not isinstance(other, _ClassAndProbFactory):
      return False
    return self.properties_to_hash == other.properties_to_hash

  def __call__(self):
    x_batch = self.model.make_input_placeholder()
    inputs = [x_batch]

    if LooseVersion(tf.__version__) < LooseVersion('1.0.0'):
      raise NotImplementedError()

    if self.attack is None:
      x_input = x_batch
    else:
      y_batch = self.model.make_label_placeholder()
      inputs.append(y_batch)
      attack_params = self.attack_params
      if attack_params is None:
        attack_params = {}
      x_input = self.attack.generate(x_batch, y=y_batch, **attack_params)

    predictions = self.model.get_probs(x_input)
    classes = tf.argmax(predictions, axis=-1)
    max_probs = tf.reduce_max(predictions, axis=1)

    return tuple(inputs), (classes, max_probs)


class _CorrectAndProbFactory(object):
  """
  A factory for an expression for the following tuple per (optionally
  adversarial) example:
    - bool per indicating whether each the example was classified correctly
    - probability assigned to that prediction
  """

  def __init__(self, model, attack=None, attack_params=None):
    if attack_params is None:
      attack_params = {}
    self.model = model
    self.attack = attack
    self.attack_params = attack_params
    hashable_attack_params = tuple((key, attack_params[key]) for key
                                   in sorted(attack_params.keys()))
    self.properties_to_hash = (model, attack, hashable_attack_params)

  def __hash__(self):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    return self.properties_to_hash.__hash__()

  def __eq__(self, other):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    if not isinstance(other, _CorrectAndProbFactory):
      return False
    return self.properties_to_hash == other.properties_to_hash

  def __call__(self):
    x_batch = self.model.make_input_placeholder()
    y_batch = self.model.make_label_placeholder()

    if LooseVersion(tf.__version__) < LooseVersion('1.0.0'):
      raise NotImplementedError()

    if self.attack is None:
      x_input = x_batch
    else:
      attack_params = self.attack_params
      if attack_params is None:
        attack_params = {}
      x_input = self.attack.generate(x_batch, y=y_batch, **attack_params)

    predictions = self.model.get_probs(x_input)
    correct = tf.equal(tf.argmax(y_batch, axis=-1),
                       tf.argmax(predictions, axis=-1))
    max_probs = tf.reduce_max(predictions, axis=1)

    return (x_batch, y_batch), (correct, max_probs)


class _AttackFactory(object):
  """
  A factory for an expression that runs an adversarial attack

  :param model: cleverhans.model.Model
  :param attack: cleverhans.attack.Attack
  :param attack_params: dict of arguments to pass to attack.generate
  :param pass_y: bool. If True, pass y to the attack.
    (Some untargeted attacks prefer to infer y to avoid label leaking.
    Targeted attacks require that y not be passed)
  """

  def __init__(self, model, attack, attack_params=None, pass_y=False):
    assert isinstance(model, cleverhans.model.Model)
    if not isinstance(attack, cleverhans.attacks.Attack):
      raise TypeError("`attack` must be an instance of cleverhans.attacks."
                      "attack. Got %s with type %s " % (str(attack),
                                                        str(type(attack))))

    if attack_params is None:
      attack_params = {}
    self.model = model
    self.attack = attack
    self.attack_params = attack_params
    self.pass_y = pass_y
    hashable_attack_params = tuple((key, attack_params[key]) for key
                                   in sorted(attack_params.keys()))
    self.properties_to_hash = (model, attack, hashable_attack_params)

  def __hash__(self):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    return self.properties_to_hash.__hash__()

  def __eq__(self, other):
    # Make factory hashable so that no two factories for the
    # same model will be used to build redundant tf graphs
    if not isinstance(other, _AttackFactory):
      return False
    return self.properties_to_hash == other.properties_to_hash

  def __call__(self):
    x_batch = self.model.make_input_placeholder()
    y_batch = self.model.make_label_placeholder()

    attack_params = self.attack_params
    if attack_params is None:
      attack_params = {}
    if self.pass_y:
      x_adv = self.attack.generate(x_batch, y=y_batch, **attack_params)
    else:
      # Some code checks the keys of kwargs, rather than checking if
      # y is None, so we need to truly not pass y at all, rather than
      # just passing a None value for it.
      x_adv = self.attack.generate(x_batch, **attack_params)

    return (x_batch, y_batch), tuple([x_adv])


_logger = create_logger("cleverhans.evaluation")

# Cache for storing output of `batch_eval_multi_worker`'s calls to
# `graph_factory`, to avoid making the tf graph too big
_batch_eval_multi_worker_cache = {}


def _check_x(x):
  """
  Makes sure an `x` argument is a valid numpy dataset.
  """
  if not isinstance(x, np.ndarray):
    raise TypeError("x must be a numpy array. Typically x contains "
                    "the entire test set inputs.")


def _check_y(y):
  """
  Makes sure a `y` argument is a vliad numpy dataset.
  """
  if not isinstance(y, np.ndarray):
    raise TypeError("y must be numpy array. Typically y contains "
                    "the entire test set labels. Got " + str(y) + " of type " + str(type(y)))
