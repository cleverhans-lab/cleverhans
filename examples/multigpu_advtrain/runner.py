"""Wrappers to TensorFlow Session.run().
"""
# pylint: disable=missing-docstring
from collections import OrderedDict


class Runner(object):
  """
  Wrap TensorFlow Session.run() by adding preprocessing and postprocessing
  steps.
  """

  def __init__(self, inputs, outputs, sess=None):
    self.sess = sess
    self.inputs = inputs
    self.outputs = outputs
    self.feed_dict = {}

  def run(self, X_batch=None):
    fetches, feed_dict = self.set_input(X_batch)
    fvals = self.sess.run(fetches, feed_dict=feed_dict)
    return self.proc_fvals(fvals)

  def set_input(self, X_batch=None):
    raise NotImplementedError('set_input not implemented.')

  def proc_fvals(self, fvals):
    raise NotImplementedError('proc_fvals not implemented.')

  def is_finished(self):
    raise NotImplementedError('is_finished not implemented.')


class RunnerMultiGPU(Runner):
  """
  Runs a graph with sub-graphs that need to run sequentially. Each sub-graph
  takes its inputs from the outputs of the previous sub-graph.
  """

  def __init__(self, *args, **kwargs):
    super(RunnerMultiGPU, self).__init__(*args, **kwargs)
    self.assert_inputs_outputs()
    self.next_vals = [None] * len(self.inputs)

  def assert_inputs_outputs(self):
    inputs = self.inputs
    outputs = self.outputs
    assert len(inputs) == len(outputs), (
        'Inputs and Outputs should match in length.')
    for i in range(len(inputs)):
      device = inputs[i].values()[0].device
      for _k, v in inputs[i].iteritems():
        assert v.device == device, (
            'Inputs should be on the same device.')
      for _k, v in outputs[i].iteritems():
        assert v.device == device, (
            'Outputs should be on the same device.')
      if i > 0:
        ikeys = inputs[i].keys()
        okeys = outputs[i-1].keys()
        # The actual requirement is looser, only the last output keys
        # should always be returned in the same order.
        assert all(ikeys[j] == okeys[j] for j in range(len(ikeys))), (
            'Inputs and outputs keys should be in the same order.')

  def set_input(self, X_batch=None):
    """
    Preprocessing the inputs before calling session.run()

    :param X_batch: A dictionary of inputs to the first sub-graph
    :return: A tuple, `(fetches, fd)`, with `fetches` being a list of
             Tensors to be fetches and `fd` the feed dictionary.
    """
    inputs = self.inputs
    outputs = self.outputs

    # data for first gpu
    fd = {}
    if X_batch is not None:
      self.next_vals[0] = OrderedDict()
      for i, vname in enumerate(self.inputs[0]):
        if vname in X_batch:
          self.next_vals[0][vname] = X_batch[vname]
        else:
          self.next_vals[0][vname] = None
    else:
      self.next_vals[0] = None

    # Set `feed_dict` for each GPU. If there is something to run for that
    # GPU, collect outputs to be fetched.
    fetches = []
    self.active_gpus = []
    for i in range(len(outputs)):
      if self.next_vals[i] is None:
        self.active_gpus += [False]
        continue
      self.active_gpus += [True]
      for k in inputs[i]:
        if self.next_vals[i][k] is not None:
          fd[inputs[i][k]] = self.next_vals[i][k]
      for k, v in outputs[i].iteritems():
        fetches += [v]

    fd.update(self.feed_dict)

    return fetches, fd

  def proc_fvals(self, fvals):
    """
    Postprocess the outputs of the Session.run(). Move the outputs of
    sub-graphs to next ones and return the output of the last sub-graph.

    :param fvals: A list of fetched values returned by Session.run()
    :return: A dictionary of fetched values returned by the last sub-graph.
    """
    inputs = self.inputs
    outputs = self.outputs

    # Move data to the next sub-graph for the next step
    cur = 0
    for i in range(len(inputs)-1):
      if not self.active_gpus[i]:
        self.next_vals[i+1] = None
        continue
      self.next_vals[i+1] = OrderedDict()
      for k in outputs[i]:
        self.next_vals[i+1][k] = fvals[cur]
        cur += 1
      if i == 0:
        self.next_vals[0] = None

    # Return the output of the last sub-graph
    last_fvals = OrderedDict()
    if self.active_gpus[-1]:
      assert cur+len(outputs[-1]) == len(fvals)
      for k in outputs[-1]:
        last_fvals[k] = fvals[cur]
        cur += 1
    return last_fvals

  def is_finished(self):
    return all(v is None for v in self.next_vals)


class RunnerSingleGPU(Runner):
  def __init__(self, *args, **kwargs):
    super(RunnerSingleGPU, self).__init__(*args, **kwargs)

  def set_input(self, X_batch=None):
    fd = {}
    for vname, v in self.inputs[0].iteritems():
      if vname in X_batch:
        fd[v] = X_batch[vname]
    fetches = self.outputs
    return fetches, fd

  def proc_fvals(self, fvals):
    """
    Nothing to post-process on single GPU.
    """
    return True

  def is_finished(self):
    """
    Single GPU trainer has no cache.
    """
    return True
