"""The LBFGS attack
"""

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_sum, softmax_cross_entropy_with_logits
from cleverhans.model import CallableModelWrapper, Model, wrapper_warning
from cleverhans import utils
from cleverhans import utils_tf

_logger = utils.create_logger("cleverhans.attacks.lbfgs")
tf_dtype = tf.as_dtype('float32')


class LBFGS(Attack):
  """
  LBFGS is the first adversarial attack for convolutional neural networks,
  and is a target & iterative attack.
  Paper link: "https://arxiv.org/pdf/1312.6199.pdf"

  :param model: cleverhans.model.Model
  :param sess: tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess, dtypestr='float32', **kwargs):
    if not isinstance(model, Model):
      wrapper_warning()
      model = CallableModelWrapper(model, 'probs')

    super(LBFGS, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('y_target',)
    self.structural_kwargs = [
        'batch_size', 'binary_search_steps', 'max_iterations',
        'initial_const', 'clip_min', 'clip_max'
    ]

  def generate(self, x, **kwargs):
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.

    :param x: (required) A tensor with the inputs.
    :param kwargs: See `parse_params`
    """
    assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
    self.parse_params(**kwargs)

    _, nb_classes = self.get_or_guess_labels(x, kwargs)

    attack = LBFGS_impl(
        self.sess, x, self.model.get_logits(x), self.y_target,
        self.binary_search_steps, self.max_iterations, self.initial_const,
        self.clip_min, self.clip_max, nb_classes, self.batch_size)

    def lbfgs_wrap(x_val, y_val):
      """
      Wrapper creating TensorFlow interface for use with py_func
      """
      return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)

    wrap = tf.py_func(lbfgs_wrap, [x, self.y_target], self.tf_dtype)
    wrap.set_shape(x.get_shape())

    return wrap

  def parse_params(self,
                   y_target=None,
                   batch_size=1,
                   binary_search_steps=5,
                   max_iterations=1000,
                   initial_const=1e-2,
                   clip_min=0,
                   clip_max=1):
    """
    :param y_target: (optional) A tensor with the one-hot target labels.
    :param batch_size: The number of inputs to include in a batch and
                       process simultaneously.
    :param binary_search_steps: The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the purturbation
                                and cross-entropy loss of classification.
    :param max_iterations: The maximum number of iterations.
    :param initial_const: The initial tradeoff-constant to use to tune the
                          relative importance of size of the perturbation
                          and cross-entropy loss of the classification.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """
    self.y_target = y_target
    self.batch_size = batch_size
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max


class LBFGS_impl(object):
  """
  Return a tensor that constructs adversarial examples for the given
  input. Generate uses tf.py_func in order to operate over tensors.

  :param sess: a TF session.
  :param x: A tensor with the inputs.
  :param logits: A tensor with model's output logits.
  :param targeted_label: A tensor with the target labels.
  :param binary_search_steps: The number of times we perform binary
                              search to find the optimal tradeoff-
                              constant between norm of the purturbation
                              and cross-entropy loss of classification.
  :param max_iterations: The maximum number of iterations.
  :param initial_const: The initial tradeoff-constant to use to tune the
                        relative importance of size of the purturbation
                        and cross-entropy loss of the classification.
  :param clip_min: Minimum input component value
  :param clip_max: Maximum input component value
  :param num_labels: The number of classes in the model's output.
  :param batch_size: Number of attacks to run simultaneously.

  """
  def __init__(self, sess, x, logits, targeted_label,
               binary_search_steps, max_iterations, initial_const, clip_min,
               clip_max, nb_classes, batch_size):
    self.sess = sess
    self.x = x
    self.logits = logits
    assert logits.op.type != 'Softmax'
    self.targeted_label = targeted_label
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.batch_size = batch_size

    self.repeat = self.binary_search_steps >= 10
    self.shape = tuple([self.batch_size] +
                       list(self.x.get_shape().as_list()[1:]))
    self.ori_img = tf.Variable(
        np.zeros(self.shape), dtype=tf_dtype, name='ori_img')
    self.const = tf.Variable(
        np.zeros(self.batch_size), dtype=tf_dtype, name='const')

    self.score = softmax_cross_entropy_with_logits(
        labels=self.targeted_label, logits=self.logits)
    self.l2dist = reduce_sum(tf.square(self.x - self.ori_img))
    # small self.const will result small adversarial perturbation
    self.loss = reduce_sum(self.score * self.const) + self.l2dist
    self.grad, = tf.gradients(self.loss, self.x)

  def attack(self, x_val, targets):
    """
    Perform the attack on the given instance for the given targets.
    """

    def lbfgs_objective(adv_x, self, targets, oimgs, CONST):
      """ returns the function value and the gradient for fmin_l_bfgs_b """
      loss = self.sess.run(
          self.loss,
          feed_dict={
              self.x: adv_x.reshape(oimgs.shape),
              self.targeted_label: targets,
              self.ori_img: oimgs,
              self.const: CONST
          })
      grad = self.sess.run(
          self.grad,
          feed_dict={
              self.x: adv_x.reshape(oimgs.shape),
              self.targeted_label: targets,
              self.ori_img: oimgs,
              self.const: CONST
          })
      return loss, grad.flatten().astype(float)

    # begin the main part for the attack
    from scipy.optimize import fmin_l_bfgs_b
    oimgs = np.clip(x_val, self.clip_min, self.clip_max)
    CONST = np.ones(self.batch_size) * self.initial_const

    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(self.batch_size)
    upper_bound = np.ones(self.batch_size) * 1e10

    # set the box constraints for the optimization function
    clip_min = self.clip_min * np.ones(oimgs.shape[:])
    clip_max = self.clip_max * np.ones(oimgs.shape[:])
    clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))

    # placeholders for the best l2 and instance attack found so far
    o_bestl2 = [1e10] * self.batch_size
    o_bestattack = np.copy(oimgs)

    for outer_step in range(self.binary_search_steps):
      _logger.debug("  Binary search step %s of %s",
                    outer_step, self.binary_search_steps)

      # The last iteration (if we run many steps) repeat the search once.
      if self.repeat and outer_step == self.binary_search_steps - 1:
        CONST = upper_bound

      # optimization function
      adv_x, _, __ = fmin_l_bfgs_b(
          lbfgs_objective,
          oimgs.flatten().astype(float),
          args=(self, targets, oimgs, CONST),
          bounds=clip_bound,
          maxiter=self.max_iterations,
          iprint=0)

      adv_x = adv_x.reshape(oimgs.shape)
      assert np.amax(adv_x) <= self.clip_max and \
          np.amin(adv_x) >= self.clip_min, \
          'fmin_l_bfgs_b returns are invalid'

      # adjust the best result (i.e., the adversarial example with the
      # smallest perturbation in terms of L_2 norm) found so far
      preds = np.atleast_1d(
          utils_tf.model_argmax(self.sess, self.x, self.logits,
                                adv_x))
      _logger.debug("predicted labels are %s", preds)

      l2s = np.zeros(self.batch_size)
      for i in range(self.batch_size):
        l2s[i] = np.sum(np.square(adv_x[i] - oimgs[i]))

      for e, (l2, pred, ii) in enumerate(zip(l2s, preds, adv_x)):
        if l2 < o_bestl2[e] and pred == np.argmax(targets[e]):
          o_bestl2[e] = l2
          o_bestattack[e] = ii

      # adjust the constant as needed
      for e in range(self.batch_size):
        if preds[e] == np.argmax(targets[e]):
          # success, divide const by two
          upper_bound[e] = min(upper_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
        else:
          # failure, either multiply by 10 if no solution found yet
          #          or do binary search with the known upper bound
          lower_bound[e] = max(lower_bound[e], CONST[e])
          if upper_bound[e] < 1e9:
            CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
          else:
            CONST[e] *= 10

      _logger.debug("  Successfully generated adversarial examples "
                    "on %s of %s instances.",
                    sum(upper_bound < 1e9), self.batch_size)
      o_bestl2 = np.array(o_bestl2)
      mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
      _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

    # return the best solution found
    o_bestl2 = np.array(o_bestl2)
    return o_bestattack
