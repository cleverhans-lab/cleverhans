"""
Functionality for evaluating expressions across entire datasets.
Includes multi-GPU support for fast evaluation.
"""

from distutils.version import LooseVersion
import numpy as np
from six.moves import range
import tensorflow as tf

from cleverhans.utils import create_logger
from cleverhans.utils_tf import infer_devices


def accuracy(sess, model, x, y, batch_size=None, devices=None, feed=None):
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
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :return: a float with the accuracy value
    """

    factory = _CorrectFactory(model)

    correct, = batch_eval_multi_worker(sess, factory, [x, y],
                                       batch_size=batch_size, devices=devices,
                                       feed=feed)

    return correct.mean()


# Cache for storing output of `batch_eval_multi_worker`'s calls to
# `graph_factory`, to avoid making the tf graph too big
_batch_eval_multi_worker_cache = {}


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
        assert numpy_inputs[i].shape[0] == m
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
        raise NotImplementedError("Your batch size is bigger than the"
                                  " dataset, this function is probably"
                                  " overkill.")

    def pad(array):
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
        for output in xrange(p):
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
        warnings.warn("`args` is deprecated and will be removde on or "
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

    def __init__(self, model):
        self.model = model

    def __hash__(self):
        # Make factory hashable so that no two factories for the
        # same model will be used to build redundant tf graphs
        return self.model.__hash__()

    def __eq__(self, other):
        # Make factory hashable so that no two factories for the
        # same model will be used to build redundant tf graphs
        if not isinstance(other, _CorrectFactory):
            return False
        return self.model == other.model

    def __call__(self):
        x_batch = self.model.make_input_placeholder()
        y_batch = self.model.make_label_placeholder()

        if LooseVersion(tf.__version__) < LooseVersion('1.0.0'):
            raise NotImplementedError()

        predictions = self.model.get_probs(x_batch)
        correct = tf.equal(tf.argmax(y_batch, axis=-1),
                           tf.argmax(predictions, axis=-1))

        return (x_batch, y_batch), (correct,)


_logger = create_logger("cleverhans.evaluation")
