# pylint: disable=missing-docstring
from collections import OrderedDict

import tensorflow as tf

from cleverhans.attacks import MadryEtAl
from cleverhans.utils_tf import clip_eta

from model import clone_variable


class MadryEtAlMultiGPU(MadryEtAl):

  """
  A multi-GPU version of the Projected Gradient Descent Attack
  (Madry et al. 2017).
  Paper link: https://arxiv.org/pdf/1706.06083.pdf

  This attack is designed to run on multiple GPUs for generating adversarial
  examples.
  Comparing to data parallelism, using this parallelization we can get
  very close to optimal n times speed up using n GPUs. The current
  implementation gets close to 6x speed up on 8 GPUs.
  """

  def __init__(self, *args, **kwargs):
    """
    Create a MadryEtAlMultiGPU instance.
    """
    super(MadryEtAlMultiGPU, self).__init__(*args, **kwargs)
    self.structural_kwargs += ['ngpu']

  def get_or_guess_labels(self, x, kwargs):
    device_name = '/gpu:0'
    self.model.set_device(device_name)
    with tf.device(device_name):
      with tf.variable_scope('model_pred'):
        ret = super(MadryEtAlMultiGPU, self).get_or_guess_labels(
            x, kwargs)
    return ret

  def attack(self, x, y_p, **kwargs):
    """
    This method creates a symoblic graph of the MadryEtAl attack on
    multiple GPUs. The graph is created on the first n GPUs.

    Stop gradient is needed to get the speed-up. This prevents us from
    being able to back-prop through the attack.

    :param x: A tensor with the input image.
    :param y_p: Ground truth label or predicted label.
    :return: Two lists containing the input and output tensors of each GPU.
    """
    inputs = []
    outputs = []

    # Create the initial random perturbation
    device_name = '/gpu:0'
    self.model.set_device(device_name)
    with tf.device(device_name):
      with tf.variable_scope('init_rand'):
        if self.rand_init:
          eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
          eta = clip_eta(eta, self.ord, self.eps)
          eta = tf.stop_gradient(eta)
        else:
          eta = tf.zeros_like(x)

    # TODO: Break the graph only nGPU times instead of nb_iter times.
    # The current implementation by the time an adversarial example is
    # used for training, the weights of the model have changed nb_iter
    # times. This can cause slower convergence compared to the single GPU
    # adversarial training.
    for i in range(self.nb_iter):
      # Create the graph for i'th step of attack
      inputs += [OrderedDict()]
      outputs += [OrderedDict()]
      device_name = x.device
      self.model.set_device(device_name)
      with tf.device(device_name):
        with tf.variable_scope('step%d' % i):
          if i > 0:
            # Clone the variables to separate the graph of 2 GPUs
            x = clone_variable('x', x)
            y_p = clone_variable('y_p', y_p)
            eta = clone_variable('eta', eta)

          inputs[i]['x'] = x
          inputs[i]['y_p'] = y_p
          outputs[i]['x'] = x
          outputs[i]['y_p'] = y_p
          inputs[i]['eta'] = eta

          eta = self.attack_single_step(x, eta, y_p)

          if i < self.nb_iter-1:
            outputs[i]['eta'] = eta
          else:
            # adv_x, not eta is the output of the last step
            adv_x = x + eta
            if (self.clip_min is not None and self.clip_max is not None):
              adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
            adv_x = tf.stop_gradient(adv_x, name='adv_x')
            outputs[i]['adv_x'] = adv_x

    return inputs, outputs

  def generate_np(self, x_val, **kwargs):
    """
    Facilitates testing this attack.
    """
    _, feedable, _feedable_types, hash_key = self.construct_variables(kwargs)

    if hash_key not in self.graphs:
      with tf.variable_scope(None, 'attack_%d' % len(self.graphs)):
        # x is a special placeholder we always want to have
        with tf.device('/gpu:0'):
          x = tf.placeholder(tf.float32, shape=x_val.shape, name='x')

        inputs, outputs = self.generate(x, **kwargs)

        from runner import RunnerMultiGPU
        runner = RunnerMultiGPU(inputs, outputs, sess=self.sess)
        self.graphs[hash_key] = runner

    runner = self.graphs[hash_key]
    feed_dict = {'x': x_val}
    for name in feedable:
      feed_dict[name] = feedable[name]
    fvals = runner.run(feed_dict)
    while not runner.is_finished():
      fvals = runner.run()

    return fvals['adv_x']

  def parse_params(self, ngpu=1, **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:
    :param ngpu: (required int) the number of GPUs available.
    :param kwargs: A dictionary of parameters for MadryEtAl attack.
    """

    return_status = super(MadryEtAlMultiGPU, self).parse_params(**kwargs)
    self.ngpu = ngpu

    return return_status
