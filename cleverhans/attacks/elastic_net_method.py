"""The ElasticNetMethod attack.
"""
# pylint: disable=missing-docstring
import logging

import numpy as np
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_sum, reduce_max
from cleverhans.model import Model, CallableModelWrapper, wrapper_warning_logits
from cleverhans import utils

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.elastic_net_method")
_logger.setLevel(logging.INFO)


def ZERO():
  return np.asarray(0., dtype=np_dtype)


class ElasticNetMethod(Attack):
  """
  This attack features L1-oriented adversarial examples and includes
  the C&W L2 attack as a special case (when beta is set to 0).
  Adversarial examples attain similar performance to those
  generated by the C&W L2 attack in the white-box case,
  and more importantly, have improved transferability properties
  and complement adversarial training.
  Paper link: https://arxiv.org/abs/1709.04114

  :param model: cleverhans.model.Model
  :param sess: tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess, dtypestr='float32', **kwargs):
    """
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
    if not isinstance(model, Model):
      wrapper_warning_logits()
      model = CallableModelWrapper(model, 'logits')

    super(ElasticNetMethod, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('y', 'y_target')

    self.structural_kwargs = [
        'beta', 'decision_rule', 'batch_size', 'confidence',
        'targeted', 'learning_rate', 'binary_search_steps',
        'max_iterations', 'abort_early', 'initial_const', 'clip_min',
        'clip_max'
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

    labels, nb_classes = self.get_or_guess_labels(x, kwargs)

    attack = EAD(self.sess, self.model, self.beta,
                 self.decision_rule, self.batch_size, self.confidence,
                 'y_target' in kwargs, self.learning_rate,
                 self.binary_search_steps, self.max_iterations,
                 self.abort_early, self.initial_const, self.clip_min,
                 self.clip_max, nb_classes,
                 x.get_shape().as_list()[1:])

    def ead_wrap(x_val, y_val):
      return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)

    wrap = tf.py_func(ead_wrap, [x, labels], self.tf_dtype)
    wrap.set_shape(x.get_shape())

    return wrap

  def parse_params(self,
                   y=None,
                   y_target=None,
                   beta=1e-2,
                   decision_rule='EN',
                   batch_size=1,
                   confidence=0,
                   learning_rate=1e-2,
                   binary_search_steps=9,
                   max_iterations=1000,
                   abort_early=False,
                   initial_const=1e-3,
                   clip_min=0,
                   clip_max=1):
    """
    :param y: (optional) A tensor with the true labels for an untargeted
              attack. If None (and y_target is None) then use the
              original labels the classifier assigns.
    :param y_target: (optional) A tensor with the target labels for a
              targeted attack.
    :param beta: Trades off L2 distortion with L1 distortion: higher
                 produces examples with lower L1 distortion, at the
                 cost of higher L2 (and typically Linf) distortion
    :param decision_rule: EN or L1. Select final adversarial example from
                          all successful examples based on the least
                          elastic-net or L1 distortion criterion.
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param batch_size: Number of attacks to run simultaneously.
    :param learning_rate: The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
    :param binary_search_steps: The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the perturbation
                                and confidence of the classification. Set
                                'initial_const' to a large value and fix
                                this param to 1 for speed.
    :param max_iterations: The maximum number of iterations. Setting this
                           to a larger value will produce lower distortion
                           results. Using only a few iterations requires
                           a larger learning rate, and will produce larger
                           distortion results.
    :param abort_early: If true, allows early abort when the total
                        loss starts to increase (greatly speeds up attack,
                        but hurts performance, particularly on ImageNet)
    :param initial_const: The initial tradeoff-constant to use to tune the
                          relative importance of size of the perturbation
                          and confidence of classification.
                          If binary_search_steps is large, the initial
                          constant is not important. A smaller value of
                          this constant gives lower distortion results.
                          For computational efficiency, fix
                          binary_search_steps to 1 and set this param
                          to a large value.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # ignore the y and y_target argument
    self.beta = beta
    self.decision_rule = decision_rule
    self.batch_size = batch_size
    self.confidence = confidence
    self.learning_rate = learning_rate
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.abort_early = abort_early
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max


class EAD(object):
  def __init__(self, sess, model, beta, decision_rule, batch_size,
               confidence, targeted, learning_rate, binary_search_steps,
               max_iterations, abort_early, initial_const, clip_min,
               clip_max, num_labels, shape):
    """
    EAD Attack

    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.

    :param sess: a TF session.
    :param model: a cleverhans.model.Model object.
    :param beta: Trades off L2 distortion with L1 distortion: higher
                 produces examples with lower L1 distortion, at the
                 cost of higher L2 (and typically Linf) distortion
    :param decision_rule: EN or L1. Select final adversarial example from
                          all successful examples based on the least
                          elastic-net or L1 distortion criterion.
    :param batch_size: Number of attacks to run simultaneously.
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param targeted: boolean controlling the behavior of the adversarial
                     examples produced. If set to False, they will be
                     misclassified in any wrong class. If set to True,
                     they will be misclassified in a chosen target class.
    :param learning_rate: The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
    :param binary_search_steps: The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the perturbation
                                and confidence of the classification. Set
                                'initial_const' to a large value and fix
                                this param to 1 for speed.
    :param max_iterations: The maximum number of iterations. Setting this
                           to a larger value will produce lower distortion
                           results. Using only a few iterations requires
                           a larger learning rate, and will produce larger
                           distortion results.
    :param abort_early: If true, allows early abort when the total
                        loss starts to increase (greatly speeds up attack,
                        but hurts performance, particularly on ImageNet)
    :param initial_const: The initial tradeoff-constant to use to tune the
                          relative importance of size of the perturbation
                          and confidence of classification.
                          If binary_search_steps is large, the initial
                          constant is not important. A smaller value of
                          this constant gives lower distortion results.
                          For computational efficiency, fix
                          binary_search_steps to 1 and set this param
                          to a large value.
    :param clip_min: (optional float) Minimum input component value.
    :param clip_max: (optional float) Maximum input component value.
    :param num_labels: the number of classes in the model's output.
    :param shape: the shape of the model's input tensor.
    """

    self.sess = sess
    self.TARGETED = targeted
    self.LEARNING_RATE = learning_rate
    self.MAX_ITERATIONS = max_iterations
    self.BINARY_SEARCH_STEPS = binary_search_steps
    self.ABORT_EARLY = abort_early
    self.CONFIDENCE = confidence
    self.initial_const = initial_const
    self.batch_size = batch_size
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.model = model
    self.decision_rule = decision_rule

    self.beta = beta
    self.beta_t = tf.cast(self.beta, tf_dtype)

    self.repeat = binary_search_steps >= 10

    self.shape = shape = tuple([batch_size] + list(shape))

    # these are variables to be more efficient in sending data to tf
    self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
    self.newimg = tf.Variable(
        np.zeros(shape), dtype=tf_dtype, name='newimg')
    self.slack = tf.Variable(
        np.zeros(shape), dtype=tf_dtype, name='slack')
    self.tlab = tf.Variable(
        np.zeros((batch_size, num_labels)), dtype=tf_dtype, name='tlab')
    self.const = tf.Variable(
        np.zeros(batch_size), dtype=tf_dtype, name='const')

    # and here's what we use to assign them
    self.assign_timg = tf.placeholder(tf_dtype, shape, name='assign_timg')
    self.assign_newimg = tf.placeholder(
        tf_dtype, shape, name='assign_newimg')
    self.assign_slack = tf.placeholder(
        tf_dtype, shape, name='assign_slack')
    self.assign_tlab = tf.placeholder(
        tf_dtype, (batch_size, num_labels), name='assign_tlab')
    self.assign_const = tf.placeholder(
        tf_dtype, [batch_size], name='assign_const')

    self.global_step = tf.Variable(0, trainable=False)
    self.global_step_t = tf.cast(self.global_step, tf_dtype)

    # Fast Iterative Shrinkage Thresholding
    self.zt = tf.divide(self.global_step_t,
                        self.global_step_t + tf.cast(3, tf_dtype))
    cond1 = tf.cast(tf.greater(tf.subtract(self.slack, self.timg),
                               self.beta_t), tf_dtype)
    cond2 = tf.cast(tf.less_equal(tf.abs(tf.subtract(self.slack,
                                                     self.timg)),
                                  self.beta_t), tf_dtype)
    cond3 = tf.cast(tf.less(tf.subtract(self.slack, self.timg),
                            tf.negative(self.beta_t)), tf_dtype)

    upper = tf.minimum(tf.subtract(self.slack, self.beta_t),
                       tf.cast(self.clip_max, tf_dtype))
    lower = tf.maximum(tf.add(self.slack, self.beta_t),
                       tf.cast(self.clip_min, tf_dtype))

    self.assign_newimg = tf.multiply(cond1, upper)
    self.assign_newimg += tf.multiply(cond2, self.timg)
    self.assign_newimg += tf.multiply(cond3, lower)

    self.assign_slack = self.assign_newimg
    self.assign_slack += tf.multiply(self.zt,
                                     self.assign_newimg - self.newimg)

    # --------------------------------
    self.setter = tf.assign(self.newimg, self.assign_newimg)
    self.setter_y = tf.assign(self.slack, self.assign_slack)

    # prediction BEFORE-SOFTMAX of the model
    self.output = model.get_logits(self.newimg)
    self.output_y = model.get_logits(self.slack)

    # distance to the input data
    self.l2dist = reduce_sum(tf.square(self.newimg-self.timg),
                             list(range(1, len(shape))))
    self.l2dist_y = reduce_sum(tf.square(self.slack-self.timg),
                               list(range(1, len(shape))))
    self.l1dist = reduce_sum(tf.abs(self.newimg-self.timg),
                             list(range(1, len(shape))))
    self.l1dist_y = reduce_sum(tf.abs(self.slack-self.timg),
                               list(range(1, len(shape))))
    self.elasticdist = self.l2dist + tf.multiply(self.l1dist,
                                                 self.beta_t)
    self.elasticdist_y = self.l2dist_y + tf.multiply(self.l1dist_y,
                                                     self.beta_t)
    if self.decision_rule == 'EN':
      self.crit = self.elasticdist
      self.crit_p = 'Elastic'
    else:
      self.crit = self.l1dist
      self.crit_p = 'L1'

    # compute the probability of the label class versus the maximum other
    real = reduce_sum((self.tlab) * self.output, 1)
    real_y = reduce_sum((self.tlab) * self.output_y, 1)
    other = reduce_max((1 - self.tlab) * self.output -
                       (self.tlab * 10000), 1)
    other_y = reduce_max((1 - self.tlab) * self.output_y -
                         (self.tlab * 10000), 1)

    if self.TARGETED:
      # if targeted, optimize for making the other class most likely
      loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
      loss1_y = tf.maximum(ZERO(), other_y - real_y + self.CONFIDENCE)
    else:
      # if untargeted, optimize for making this class least likely.
      loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)
      loss1_y = tf.maximum(ZERO(), real_y - other_y + self.CONFIDENCE)

    # sum up the losses
    self.loss21 = reduce_sum(self.l1dist)
    self.loss21_y = reduce_sum(self.l1dist_y)
    self.loss2 = reduce_sum(self.l2dist)
    self.loss2_y = reduce_sum(self.l2dist_y)
    self.loss1 = reduce_sum(self.const * loss1)
    self.loss1_y = reduce_sum(self.const * loss1_y)
    self.loss_opt = self.loss1_y + self.loss2_y
    self.loss = self.loss1+self.loss2+tf.multiply(self.beta_t, self.loss21)

    self.learning_rate = tf.train.polynomial_decay(
        self.LEARNING_RATE,
        self.global_step,
        self.MAX_ITERATIONS,
        0,
        power=0.5)

    # Setup the optimizer and keep track of variables we're creating
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train = optimizer.minimize(self.loss_opt,
                                    var_list=[self.slack],
                                    global_step=self.global_step)
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]

    # these are the variables to initialize when we run
    self.setup = []
    self.setup.append(self.timg.assign(self.assign_timg))
    self.setup.append(self.tlab.assign(self.assign_tlab))
    self.setup.append(self.const.assign(self.assign_const))

    var_list = [self.global_step]+[self.slack]+[self.newimg]+new_vars
    self.init = tf.variables_initializer(var_list=var_list)

  def attack(self, imgs, targets):
    """
    Perform the EAD attack on the given instance for the given targets.

    If self.targeted is true, then the targets represents the target labels
    If self.targeted is false, then targets are the original class labels
    """

    batch_size = self.batch_size
    r = []
    for i in range(0, len(imgs) // batch_size):
      _logger.debug(
          ("Running EAD attack on instance %s of %s",
           i * batch_size, len(imgs)))
      r.extend(
          self.attack_batch(
              imgs[i * batch_size:(i + 1) * batch_size],
              targets[i * batch_size:(i + 1) * batch_size]))
    if len(imgs) % batch_size != 0:
      last_elements = len(imgs) - (len(imgs) % batch_size)
      _logger.debug(
          ("Running EAD attack on instance %s of %s",
           last_elements, len(imgs)))
      temp_imgs = np.zeros((batch_size, ) + imgs.shape[2:])
      temp_targets = np.zeros((batch_size, ) + targets.shape[2:])
      temp_imgs[:(len(imgs) % batch_size)] = imgs[last_elements:]
      temp_targets[:(len(imgs) % batch_size)] = targets[last_elements:]
      temp_data = self.attack_batch(temp_imgs, temp_targets)
      r.extend(temp_data[:(len(imgs) % batch_size)],
               targets[last_elements:])
    return np.array(r)

  def attack_batch(self, imgs, labs):
    """
    Run the attack on a batch of instance and labels.
    """

    def compare(x, y):
      if not isinstance(x, (float, int, np.int64)):
        x = np.copy(x)
        if self.TARGETED:
          x[y] -= self.CONFIDENCE
        else:
          x[y] += self.CONFIDENCE
        x = np.argmax(x)
      if self.TARGETED:
        return x == y
      else:
        return x != y

    batch_size = self.batch_size

    imgs = np.clip(imgs, self.clip_min, self.clip_max)

    # set the lower and upper bounds accordingly
    lower_bound = np.zeros(batch_size)
    CONST = np.ones(batch_size) * self.initial_const
    upper_bound = np.ones(batch_size) * 1e10

    # placeholders for the best en, score, and instance attack found so far
    o_bestdst = [1e10] * batch_size
    o_bestscore = [-1] * batch_size
    o_bestattack = np.copy(imgs)

    for outer_step in range(self.BINARY_SEARCH_STEPS):
      # completely reset the optimizer's internal state.
      self.sess.run(self.init)
      batch = imgs[:batch_size]
      batchlab = labs[:batch_size]

      bestdst = [1e10] * batch_size
      bestscore = [-1] * batch_size
      _logger.debug("  Binary search step %s of %s",
                    outer_step, self.BINARY_SEARCH_STEPS)

      # The last iteration (if we run many steps) repeat the search once.
      if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
        CONST = upper_bound

      # set the variables so that we don't have to send them over again
      self.sess.run(
          self.setup, {
              self.assign_timg: batch,
              self.assign_tlab: batchlab,
              self.assign_const: CONST
          })
      self.sess.run(self.setter, {self.assign_newimg: batch})
      self.sess.run(self.setter_y, {self.assign_slack: batch})
      prev = 1e6
      for iteration in range(self.MAX_ITERATIONS):
        # perform the attack
        self.sess.run([self.train])
        self.sess.run([self.setter, self.setter_y])
        l, l2s, l1s, crit, scores, nimg = self.sess.run([self.loss,
                                                         self.l2dist,
                                                         self.l1dist,
                                                         self.crit,
                                                         self.output,
                                                         self.newimg])
        if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
          _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                         "l2={:.3g} l1={:.3g} f={:.3g}").format(
                             iteration, self.MAX_ITERATIONS, l,
                             np.mean(l2s), np.mean(l1s),
                             np.mean(scores)))

        # check if we should abort search if we're getting nowhere.
        if self.ABORT_EARLY and \
           iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
          if l > prev * .9999:
            msg = "    Failed to make progress; stop early"
            _logger.debug(msg)
            break
          prev = l

        # adjust the best result found so far
        for e, (dst, sc, ii) in enumerate(zip(crit, scores, nimg)):
          lab = np.argmax(batchlab[e])
          if dst < bestdst[e] and compare(sc, lab):
            bestdst[e] = dst
            bestscore[e] = np.argmax(sc)
          if dst < o_bestdst[e] and compare(sc, lab):
            o_bestdst[e] = dst
            o_bestscore[e] = np.argmax(sc)
            o_bestattack[e] = ii

      # adjust the constant as needed
      for e in range(batch_size):
        if compare(bestscore[e], np.argmax(batchlab[e])) and \
           bestscore[e] != -1:
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
      _logger.debug("  Successfully generated adversarial examples " +
                    "on {} of {} instances.".format(
                        sum(upper_bound < 1e9), batch_size))
      o_bestdst = np.array(o_bestdst)
      mean = np.mean(np.sqrt(o_bestdst[o_bestdst < 1e9]))
      _logger.debug(self.crit_p +
                    " Mean successful distortion: {:.4g}".format(mean))

    # return the best solution found
    o_bestdst = np.array(o_bestdst)
    return o_bestattack
