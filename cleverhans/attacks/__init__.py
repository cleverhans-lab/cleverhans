from abc import ABCMeta
import collections
import warnings
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans import utils
from cleverhans.attacks.attack import Attack
from cleverhans.attacks.basic_iterative_method import BasicIterativeMethod
from cleverhans.attacks.fast_gradient_method import FastGradientMethod, fgm, optimize_linear
from cleverhans.attacks.madry_et_al import MadryEtAl
from cleverhans.attacks.projected_gradient_descent import ProjectedGradientDescent
from cleverhans.attacks_tf import SPSAAdam, margin_logit_loss, TensorAdam
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.model import wrapper_warning, wrapper_warning_logits
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans.utils_tf import clip_eta
from cleverhans import utils_tf

_logger = utils.create_logger("cleverhans.attacks")
tf_dtype = tf.as_dtype('float32')


class MomentumIterativeMethod(Attack):
  """
  The Momentum Iterative Method (Dong et al. 2017). This method won
  the first places in NIPS 2017 Non-targeted Adversarial Attacks and
  Targeted Adversarial Attacks. The original paper used hard labels
  for this attack; no label smoothing.
  Paper link: https://arxiv.org/pdf/1710.06081.pdf

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a MomentumIterativeMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(MomentumIterativeMethod, self).__init__(model, sess, dtypestr,
                                                  **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                            'clip_max')
    self.structural_kwargs = [
        'ord', 'nb_iter', 'decay_factor', 'sanity_checks']

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: Keyword arguments. See `parse_params` for documentation.
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    asserts = []

    # If a data range was specified, check that the input was in that range
    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(x,
                                                   tf.cast(self.clip_min,
                                                           x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(x,
                                                tf.cast(self.clip_max,
                                                        x.dtype)))

    # Initialize loop variables
    momentum = tf.zeros_like(x)
    adv_x = x

    # Fix labels to the first model predictions for loss computation
    y, _nb_classes = self.get_or_guess_labels(x, kwargs)
    y = y / reduce_sum(y, 1, keepdims=True)
    targeted = (self.y_target is not None)

    def cond(i, _, __):
      return tf.less(i, self.nb_iter)

    def body(i, ax, m):
      logits = self.model.get_logits(ax)
      loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
      if targeted:
        loss = -loss

      # Define gradient of loss wrt input
      grad, = tf.gradients(loss, ax)

      # Normalize current gradient and add it to the accumulated gradient
      red_ind = list(xrange(1, len(grad.get_shape())))
      avoid_zero_div = tf.cast(1e-12, grad.dtype)
      grad = grad / tf.maximum(
          avoid_zero_div,
          reduce_mean(tf.abs(grad), red_ind, keepdims=True))
      m = self.decay_factor * m + grad

      optimal_perturbation = optimize_linear(m, self.eps_iter, self.ord)
      if self.ord == 1:
        raise NotImplementedError("This attack hasn't been tested for ord=1."
                                  "It's not clear that FGM makes a good inner "
                                  "loop step for iterative optimization since "
                                  "it updates just one coordinate at a time.")

      # Update and clip adversarial example in current iteration
      ax = ax + optimal_perturbation
      ax = x + utils_tf.clip_eta(ax - x, self.ord, self.eps)

      if self.clip_min is not None and self.clip_max is not None:
        ax = utils_tf.clip_by_value(ax, self.clip_min, self.clip_max)

      ax = tf.stop_gradient(ax)

      return i + 1, ax, m

    _, adv_x, _ = tf.while_loop(
        cond, body, (tf.zeros([]), adv_x, momentum), back_prop=True,
        maximum_iterations=self.nb_iter)

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

  def parse_params(self,
                   eps=0.3,
                   eps_iter=0.06,
                   nb_iter=10,
                   y=None,
                   ord=np.inf,
                   decay_factor=1.0,
                   clip_min=None,
                   clip_max=None,
                   y_target=None,
                   sanity_checks=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the true labels.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param decay_factor: (optional) Decay factor for the momentum term.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # Save attack-specific parameters
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.decay_factor = decay_factor
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.sanity_checks = sanity_checks

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class SaliencyMapMethod(Attack):
  """
  The Jacobian-based Saliency Map Method (Papernot et al. 2016).
  Paper link: https://arxiv.org/pdf/1511.07528.pdf

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor

  :note: When not using symbolic implementation in `generate`, `sess` should
         be provided
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a SaliencyMapMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(SaliencyMapMethod, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('y_target',)
    self.structural_kwargs = [
        'theta', 'gamma', 'clip_max', 'clip_min', 'symbolic_impl'
    ]

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    if self.symbolic_impl:
      from cleverhans.attacks_tf import jsma_symbolic

      # Create random targets if y_target not provided
      if self.y_target is None:
        from random import randint

        def random_targets(gt):
          result = gt.copy()
          nb_s = gt.shape[0]
          nb_classes = gt.shape[1]

          for i in xrange(nb_s):
            result[i, :] = np.roll(result[i, :],
                                   randint(1, nb_classes - 1))

          return result

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.y_target = tf.py_func(random_targets, [labels],
                                   self.tf_dtype)
        self.y_target.set_shape([None, nb_classes])

      x_adv = jsma_symbolic(
          x,
          model=self.model,
          y_target=self.y_target,
          theta=self.theta,
          gamma=self.gamma,
          clip_min=self.clip_min,
          clip_max=self.clip_max)
    else:
      assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
      from cleverhans.attacks_tf import jacobian_graph, jsma_batch

      # Define Jacobian graph wrt to this input placeholder
      preds = self.model.get_probs(x)
      nb_classes = preds.get_shape().as_list()[-1]
      grads = jacobian_graph(preds, x, nb_classes)

      # Define appropriate graph (targeted / random target labels)
      if self.y_target is not None:

        def jsma_wrap(x_val, y_target):
          return jsma_batch(
              self.sess,
              x,
              preds,
              grads,
              x_val,
              self.theta,
              self.gamma,
              self.clip_min,
              self.clip_max,
              nb_classes,
              y_target=y_target)

        # Attack is targeted, target placeholder will need to be fed
        x_adv = tf.py_func(jsma_wrap, [x, self.y_target],
                           self.tf_dtype)
      else:

        def jsma_wrap(x_val):
          return jsma_batch(
              self.sess,
              x,
              preds,
              grads,
              x_val,
              self.theta,
              self.gamma,
              self.clip_min,
              self.clip_max,
              nb_classes,
              y_target=None)

        # Attack is untargeted, target values will be chosen at random
        x_adv = tf.py_func(jsma_wrap, [x], self.tf_dtype)
        x_adv.set_shape(x.get_shape())

    return x_adv

  def parse_params(self,
                   theta=1.,
                   gamma=1.,
                   clip_min=0.,
                   clip_max=1.,
                   y_target=None,
                   symbolic_impl=True,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param theta: (optional float) Perturbation introduced to modified
                  components (can be positive or negative)
    :param gamma: (optional float) Maximum percentage of perturbed features
    :param clip_min: (optional float) Minimum component value for clipping
    :param clip_max: (optional float) Maximum component value for clipping
    :param y_target: (optional) Target tensor if the attack is targeted
    """
    self.theta = theta
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.y_target = y_target
    self.symbolic_impl = symbolic_impl

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class VirtualAdversarialMethod(Attack):
  """
  This attack was originally proposed by Miyato et al. (2016) and was used
  for virtual adversarial training.
  Paper link: https://arxiv.org/abs/1507.00677

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
    if not isinstance(model, Model):
      wrapper_warning_logits()
      model = CallableModelWrapper(model, 'logits')

    super(VirtualAdversarialMethod, self).__init__(model, sess, dtypestr,
                                                   **kwargs)

    self.feedable_kwargs = ('eps', 'xi', 'clip_min', 'clip_max')
    self.structural_kwargs = ['num_iterations']

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    return vatm(
        self.model,
        x,
        self.model.get_logits(x),
        eps=self.eps,
        num_iterations=self.num_iterations,
        xi=self.xi,
        clip_min=self.clip_min,
        clip_max=self.clip_max)

  def parse_params(self,
                   eps=2.0,
                   nb_iter=None,
                   xi=1e-6,
                   clip_min=None,
                   clip_max=None,
                   num_iterations=None,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float )the epsilon (input variation parameter)
    :param nb_iter: (optional) the number of iterations
      Defaults to 1 if not specified
    :param xi: (optional float) the finite difference parameter
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param num_iterations: Deprecated alias for `nb_iter`
    """
    # Save attack-specific parameters
    self.eps = eps
    if num_iterations is not None:
      warnings.warn("`num_iterations` is deprecated. Switch to `nb_iter`."
                    " The old name will be removed on or after 2019-04-26.")
      # Note: when we remove the deprecated alias, we can put the default
      # value of 1 for nb_iter back in the method signature
      assert nb_iter is None
      nb_iter = num_iterations
    del num_iterations
    if nb_iter is None:
      nb_iter = 1
    self.num_iterations = nb_iter
    self.xi = xi
    self.clip_min = clip_min
    self.clip_max = clip_max
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")
    return True


class CarliniWagnerL2(Attack):
  """
  This attack was originally proposed by Carlini and Wagner. It is an
  iterative attack that finds adversarial examples on many defenses that
  are robust to other attacks.
  Paper link: https://arxiv.org/abs/1608.04644

  At a high level, this attack is an iterative attack using Adam and
  a specially-chosen loss function to find adversarial examples with
  lower distortion than other attacks. This comes at the cost of speed,
  as this attack is often much slower than others.

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

    super(CarliniWagnerL2, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('y', 'y_target')

    self.structural_kwargs = [
        'batch_size', 'confidence', 'targeted', 'learning_rate',
        'binary_search_steps', 'max_iterations', 'abort_early',
        'initial_const', 'clip_min', 'clip_max'
    ]

  def generate(self, x, **kwargs):
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.

    :param x: A tensor with the inputs.
    :param kwargs: See `parse_params`
    """
    assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
    from cleverhans.attacks_tf import CarliniWagnerL2 as CWL2
    self.parse_params(**kwargs)

    labels, nb_classes = self.get_or_guess_labels(x, kwargs)

    attack = CWL2(self.sess, self.model, self.batch_size, self.confidence,
                  'y_target' in kwargs, self.learning_rate,
                  self.binary_search_steps, self.max_iterations,
                  self.abort_early, self.initial_const, self.clip_min,
                  self.clip_max, nb_classes,
                  x.get_shape().as_list()[1:])

    def cw_wrap(x_val, y_val):
      return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)

    wrap = tf.py_func(cw_wrap, [x, labels], self.tf_dtype)
    wrap.set_shape(x.get_shape())

    return wrap

  def parse_params(self,
                   y=None,
                   y_target=None,
                   batch_size=1,
                   confidence=0,
                   learning_rate=5e-3,
                   binary_search_steps=5,
                   max_iterations=1000,
                   abort_early=True,
                   initial_const=1e-2,
                   clip_min=0,
                   clip_max=1):
    """
    :param y: (optional) A tensor with the true labels for an untargeted
              attack. If None (and y_target is None) then use the
              original labels the classifier assigns.
    :param y_target: (optional) A tensor with the target labels for a
              targeted attack.
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param batch_size: Number of attacks to run simultaneously.
    :param learning_rate: The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
    :param binary_search_steps: The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the purturbation
                                and confidence of the classification.
    :param max_iterations: The maximum number of iterations. Setting this
                           to a larger value will produce lower distortion
                           results. Using only a few iterations requires
                           a larger learning rate, and will produce larger
                           distortion results.
    :param abort_early: If true, allows early aborts if gradient descent
                        is unable to make progress (i.e., gets stuck in
                        a local minimum).
    :param initial_const: The initial tradeoff-constant to use to tune the
                          relative importance of size of the perturbation
                          and confidence of classification.
                          If binary_search_steps is large, the initial
                          constant is not important. A smaller value of
                          this constant gives lower distortion results.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # ignore the y and y_target argument
    self.batch_size = batch_size
    self.confidence = confidence
    self.learning_rate = learning_rate
    self.binary_search_steps = binary_search_steps
    self.max_iterations = max_iterations
    self.abort_early = abort_early
    self.initial_const = initial_const
    self.clip_min = clip_min
    self.clip_max = clip_max


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

    from cleverhans.attacks_tf import ElasticNetMethod as EAD
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


class DeepFool(Attack):
  """
  DeepFool is an untargeted & iterative attack which is based on an
  iterative linearization of the classifier. The implementation here
  is w.r.t. the L2 norm.
  Paper link: "https://arxiv.org/pdf/1511.04599.pdf"

  :param model: cleverhans.model.Model
  :param sess: tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess, dtypestr='float32', **kwargs):
    """
    Create a DeepFool instance.
    """
    if not isinstance(model, Model):
      wrapper_warning_logits()
      model = CallableModelWrapper(model, 'logits')

    super(DeepFool, self).__init__(model, sess, dtypestr, **kwargs)

    self.structural_kwargs = [
        'overshoot', 'max_iter', 'clip_max', 'clip_min', 'nb_candidate'
    ]

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    assert self.sess is not None, \
      'Cannot use `generate` when no `sess` was provided'
    from cleverhans.attacks_tf import jacobian_graph, deepfool_batch

    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    # Define graph wrt to this input placeholder
    logits = self.model.get_logits(x)
    self.nb_classes = logits.get_shape().as_list()[-1]
    assert self.nb_candidate <= self.nb_classes, \
        'nb_candidate should not be greater than nb_classes'
    preds = tf.reshape(
        tf.nn.top_k(logits, k=self.nb_candidate)[0],
        [-1, self.nb_candidate])
    # grads will be the shape [batch_size, nb_candidate, image_size]
    grads = tf.stack(jacobian_graph(preds, x, self.nb_candidate), axis=1)

    # Define graph
    def deepfool_wrap(x_val):
      return deepfool_batch(self.sess, x, preds, logits, grads, x_val,
                            self.nb_candidate, self.overshoot,
                            self.max_iter, self.clip_min, self.clip_max,
                            self.nb_classes)

    wrap = tf.py_func(deepfool_wrap, [x], self.tf_dtype)
    wrap.set_shape(x.get_shape())
    return wrap

  def parse_params(self,
                   nb_candidate=10,
                   overshoot=0.02,
                   max_iter=50,
                   clip_min=0.,
                   clip_max=1.,
                   **kwargs):
    """
    :param nb_candidate: The number of classes to test against, i.e.,
                         deepfool only consider nb_candidate classes when
                         attacking(thus accelerate speed). The nb_candidate
                         classes are chosen according to the prediction
                         confidence during implementation.
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for deepfool
    :param clip_min: Minimum component value for clipping
    :param clip_max: Maximum component value for clipping
    """
    self.nb_candidate = nb_candidate
    self.overshoot = overshoot
    self.max_iter = max_iter
    self.clip_min = clip_min
    self.clip_max = clip_max
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


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
    """
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
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
  def __init__(self, sess, x, logits, targeted_label,
               binary_search_steps, max_iterations, initial_const, clip_min,
               clip_max, nb_classes, batch_size):
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
      # returns the function value and the gradient for fmin_l_bfgs_b
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


def vatm(model,
         x,
         logits,
         eps,
         back='tf',
         num_iterations=1,
         xi=1e-6,
         clip_min=None,
         clip_max=None):
  """
  A wrapper for the perturbation methods used for virtual adversarial
  training : https://arxiv.org/abs/1507.00677
  It calls the right function, depending on the
  user's backend.

  :param model: the model which returns the network unnormalized logits
  :param x: the input placeholder
  :param logits: the model's unnormalized output tensor
  :param eps: the epsilon (input variation parameter)
  :param num_iterations: the number of iterations
  :param xi: the finite difference parameter
  :param clip_min: optional parameter that can be used to set a minimum
                  value for components of the example returned
  :param clip_max: optional parameter that can be used to set a maximum
                  value for components of the example returned
  :return: a tensor for the adversarial example

  """
  assert back == 'tf'
  # Compute VATM using TensorFlow
  from cleverhans.attacks_tf import vatm as vatm_tf
  return vatm_tf(
      model,
      x,
      logits,
      eps,
      num_iterations=num_iterations,
      xi=xi,
      clip_min=clip_min,
      clip_max=clip_max)


class FastFeatureAdversaries(Attack):
  """
  This is a fast implementation of "Feature Adversaries", an attack
  against a target internal representation of a model.
  "Feature adversaries" were originally introduced in (Sabour et al. 2016),
  where the optimization was done using LBFGS.
  Paper link: https://arxiv.org/abs/1511.05122

  This implementation is similar to "Basic Iterative Method"
  (Kurakin et al. 2016) but applied to the internal representations.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a FastFeatureAdversaries instance.
    """
    super(FastFeatureAdversaries, self).__init__(model, sess, dtypestr,
                                                 **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'clip_min', 'clip_max',
                            'layer')
    self.structural_kwargs = ['ord', 'nb_iter']

    assert isinstance(self.model, Model)

  def parse_params(self,
                   layer=None,
                   eps=0.3,
                   eps_iter=0.05,
                   nb_iter=10,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param layer: (required str) name of the layer to target.
    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # Save attack-specific parameters
    self.layer = layer
    self.eps = eps
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True

  def attack_single_step(self, x, eta, g_feat):
    """
    TensorFlow implementation of the Fast Feature Gradient. This is a
    single step attack similar to Fast Gradient Method that attacks an
    internal representation.

    :param x: the input placeholder
    :param eta: A tensor the same shape as x that holds the perturbation.
    :param g_feat: model's internal tensor for guide
    :return: a tensor for the adversarial example
    """

    adv_x = x + eta
    a_feat = self.model.fprop(adv_x)[self.layer]

    # feat.shape = (batch, c) or (batch, w, h, c)
    axis = list(range(1, len(a_feat.shape)))

    # Compute loss
    # This is a targeted attack, hence the negative sign
    loss = -reduce_sum(tf.square(a_feat - g_feat), axis)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, adv_x)

    # Multiply by constant epsilon
    scaled_signed_grad = self.eps_iter * tf.sign(grad)

    # Add perturbation to original example to obtain adversarial example
    adv_x = adv_x + scaled_signed_grad

    # If clipping is needed,
    # reset all values outside of [clip_min, clip_max]
    if (self.clip_min is not None) and (self.clip_max is not None):
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    adv_x = tf.stop_gradient(adv_x)

    eta = adv_x - x
    eta = clip_eta(eta, self.ord, self.eps)

    return eta

  def generate(self, x, g, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param g: The target value of the symbolic representation
    :param kwargs: See `parse_params`
    """

    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    g_feat = self.model.fprop(g)[self.layer]

    # Initialize loop variables
    eta = tf.random_uniform(
        tf.shape(x), -self.eps, self.eps, dtype=self.tf_dtype)
    eta = clip_eta(eta, self.ord, self.eps)

    def cond(i, _):
      return tf.less(i, self.nb_iter)

    def body(i, e):
      new_eta = self.attack_single_step(x, e, g_feat)
      return i + 1, new_eta

    _, eta = tf.while_loop(cond, body, (tf.zeros([]), eta), back_prop=True,
                           maximum_iterations=self.nb_iter)

    # Define adversarial example (and clip if necessary)
    adv_x = x + eta
    if self.clip_min is not None and self.clip_max is not None:
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    return adv_x


class SPSA(Attack):
  """
  This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
  (Uesato et al. 2018). SPSA is a gradient-free optimization method, which
  is useful when the model is non-differentiable, or more generally, the
  gradients do not point in useful directions.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  DEFAULT_SPSA_SAMPLES = 128
  DEFAULT_SPSA_ITERS = 1
  DEFAULT_DELTA = 0.01
  DEFAULT_LEARNING_RATE = 0.01

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(SPSA, self).__init__(model, sess, dtypestr, **kwargs)

    self.feedable_kwargs = ('eps', 'clip_min', 'clip_max', 'y', 'y_target')
    self.structural_kwargs = [
        'nb_iter',
        'spsa_samples',
        'spsa_iters',
        'early_stop_loss_threshold',
        'is_debug',
        'is_targeted',
    ]

    assert isinstance(self.model, Model)

  def generate(self,
               x,
               y=None,
               y_target=None,
               eps=None,
               clip_min=None,
               clip_max=None,
               nb_iter=None,
               is_targeted=None,
               early_stop_loss_threshold=None,
               learning_rate=DEFAULT_LEARNING_RATE,
               delta=DEFAULT_DELTA,
               spsa_samples=DEFAULT_SPSA_SAMPLES,
               batch_size=None,
               spsa_iters=DEFAULT_SPSA_ITERS,
               is_debug=False,
               epsilon=None,
               num_steps=None):
    """
    Generate symbolic graph for adversarial examples.

    :param x: The model's symbolic inputs. Must be a batch of size 1.
    :param y: A Tensor or None. The index of the correct label.
    :param y_target: A Tensor or None. The index of the target label in a
                     targeted attack.
    :param eps: The size of the maximum perturbation, measured in the
                L-infinity norm.
    :param clip_min: If specified, the minimum input value
    :param clip_max: If specified, the maximum input value
    :param nb_iter: The number of optimization steps.
    :param early_stop_loss_threshold: A float or None. If specified, the
                                      attack will end as soon as the loss
                                      is below `early_stop_loss_threshold`.
    :param learning_rate: Learning rate of ADAM optimizer.
    :param delta: Perturbation size used for SPSA approximation.
    :param spsa_samples: Number of inputs to evaluate at a single time.
                       The true batch size (the number of evaluated
                       inputs for each update) is `spsa_samples *
                       spsa_iters`
    :param batch_size: Deprecated param that is an alias for spsa_samples
    :param spsa_iters: Number of model evaluations before performing an
                       update, where each evaluation is on `spsa_samples`
                       different inputs.
    :param is_debug: If True, print the adversarial loss after each update.
    :param epsilon: Deprecated alias for `eps`
    :param num_steps: Deprecated alias for `nb_iter`.
    :param is_targeted: Deprecated argument. Ignored.
    """

    if epsilon is not None:
      if eps is not None:
        raise ValueError("Should not specify both eps and its deprecated "
                         "alias, epsilon")
      warnings.warn("`epsilon` is deprecated. Switch to `eps`. `epsilon` may "
                    "be removed on or after 2019-04-15.")
      eps = epsilon
    del epsilon

    if num_steps is not None:
      if nb_iter is not None:
        raise ValueError("Should not specify both nb_iter and its deprecated "
                         "alias, num_steps")
      warnings.warn("`num_steps` is deprecated. Switch to `nb_iter`. "
                    "`num_steps` may be removed on or after 2019-04-15.")
      nb_iter = num_steps
    del num_steps
    assert nb_iter is not None

    if (y is not None) + (y_target is not None) != 1:
      raise ValueError("Must specify exactly one of y (untargeted attack, "
                       "cause the input not to be classified as this true "
                       "label) and y_target (targeted attack, cause the "
                       "input to be classified as this target label).")

    if is_targeted is not None:
      warnings.warn("`is_targeted` is deprecated. Simply do not specify it."
                    " It may become an error to specify it on or after "
                    "2019-04-15.")
      assert is_targeted == y_target is not None

    is_targeted = y_target is not None

    if x.get_shape().as_list()[0] is None:
      check_batch = utils_tf.assert_equal(tf.shape(x)[0], 1)
      with tf.control_dependencies([check_batch]):
        x = tf.identity(x)
    elif x.get_shape().as_list()[0] != 1:
      raise ValueError("For SPSA, input tensor x must have batch_size of 1.")

    if batch_size is not None:
      warnings.warn(
          'The "batch_size" argument to SPSA is deprecated, and will '
          'be removed on 2019-03-17. '
          'Please use spsa_samples instead.')
      spsa_samples = batch_size

    optimizer = SPSAAdam(
        lr=learning_rate,
        delta=delta,
        num_samples=spsa_samples,
        num_iters=spsa_iters)

    def loss_fn(x, label):
      logits = self.model.get_logits(x)
      loss_multiplier = 1 if is_targeted else -1
      return loss_multiplier * margin_logit_loss(
          logits, label,
          nb_classes=self.model.nb_classes or logits.get_shape()[-1])

    y_attack = y_target if is_targeted else y
    adv_x = projected_optimization(
        loss_fn,
        x,
        y_attack,
        eps,
        num_steps=nb_iter,
        optimizer=optimizer,
        early_stop_loss_threshold=early_stop_loss_threshold,
        is_debug=is_debug,
        clip_min=clip_min,
        clip_max=clip_max
    )
    return adv_x

  def generate_np(self, x_val, **kwargs):
    if "epsilon" in kwargs:
      warnings.warn("Using deprecated argument: see `generate`")
      assert "eps" not in kwargs
      kwargs["eps"] = kwargs["epsilon"]
      del kwargs["epsilon"]
    assert "eps" in kwargs

    if "num_steps" in kwargs:
      warnings.warn("Using deprecated argument: see `generate`")
      assert "nb_iter" not in kwargs
      kwargs["nb_iter"] = kwargs["num_steps"]
      del kwargs["num_steps"]

    if 'y' in kwargs and kwargs['y'] is not None:
      assert kwargs['y'].dtype == np.int32
    if 'y_target' in kwargs and kwargs['y_target'] is not None:
      assert kwargs['y_target'].dtype == np.int32

    # Call self.generate() sequentially for each image in the batch
    x_adv = []
    batch_size = x_val.shape[0]
    y = kwargs.pop('y', [None] * batch_size)
    assert len(x_val) == len(y), '# of images and labels should match'
    for x_single, y_single in zip(x_val, y):
      x = np.expand_dims(x_single, axis=0)
      adv_img = super(SPSA, self).generate_np(x, y=y_single, **kwargs)
      x_adv.append(adv_img)
    return np.concatenate(x_adv, axis=0)


class SpatialTransformationMethod(Attack):
  """
  Spatial transformation attack
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a SpatialTransformationMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
    """

    super(SpatialTransformationMethod, self).__init__(
        model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('n_samples', 'dx_min', 'dx_max', 'n_dxs', 'dy_min',
                            'dy_max', 'n_dys', 'angle_min', 'angle_max',
                            'n_angles', 'black_border_size')

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    from cleverhans.attacks_tf import spm

    labels, _ = self.get_or_guess_labels(x, kwargs)

    return spm(
        x,
        self.model,
        y=labels,
        n_samples=self.n_samples,
        dx_min=self.dx_min, dx_max=self.dx_max, n_dxs=self.n_dxs,
        dy_min=self.dy_min, dy_max=self.dy_max, n_dys=self.n_dys,
        angle_min=self.angle_min, angle_max=self.angle_max,
        n_angles=self.n_angles, black_border_size=self.black_border_size)

  def parse_params(self,
                   n_samples=None,
                   dx_min=-0.1,
                   dx_max=0.1,
                   n_dxs=2,
                   dy_min=-0.1,
                   dy_max=0.1,
                   n_dys=2,
                   angle_min=-30,
                   angle_max=30,
                   n_angles=6,
                   black_border_size=0,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    :param n_samples: (optional) The number of transformations sampled to
                      construct the attack. Set it to None to run
                      full grid attack.
    :param dx_min: (optional float) Minimum translation ratio along x-axis.
    :param dx_max: (optional float) Maximum translation ratio along x-axis.
    :param n_dxs: (optional int) Number of discretized translation ratios
                  along x-axis.
    :param dy_min: (optional float) Minimum translation ratio along y-axis.
    :param dy_max: (optional float) Maximum translation ratio along y-axis.
    :param n_dys: (optional int) Number of discretized translation ratios
                  along y-axis.
    :param angle_min: (optional float) Largest counter-clockwise rotation
                      angle.
    :param angle_max: (optional float) Largest clockwise rotation angle.
    :param n_angles: (optional int) Number of discretized angles.
    :param black_border_size: (optional int) size of the black border in pixels.
    """
    self.n_samples = n_samples
    self.dx_min = dx_min
    self.dx_max = dx_max
    self.n_dxs = n_dxs
    self.dy_min = dy_min
    self.dy_max = dy_max
    self.n_dys = n_dys
    self.angle_min = angle_min
    self.angle_max = angle_max
    self.n_angles = n_angles
    self.black_border_size = black_border_size

    if self.dx_min < -1 or self.dy_min < -1 or \
       self.dx_max > 1 or self.dy_max > 1:
      raise ValueError("The value of translation must be bounded "
                       "within [-1, 1]")
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")
    return True


class Semantic(Attack):
  """
  Semantic adversarial examples

  https://arxiv.org/abs/1703.06857

  Note: data must either be centered (so that the negative image can be
  made by simple negation) or must be in the interval [-1, 1]

  :param model: cleverhans.model.Model
  :param center: bool
    If True, assumes data has 0 mean so the negative image is just negation.
    If False, assumes data is in the interval [0, max_val]
  :param max_val: float
    Maximum value allowed in the input data
  :param sess: optional tf.Session
  :param dtypestr: dtype of data
  :param kwargs: passed through to the super constructor
  """

  def __init__(self, model, center, max_val=1., sess=None, dtypestr='float32',
               **kwargs):
    super(Semantic, self).__init__(model, sess, dtypestr, **kwargs)
    self.center = center
    self.max_val = max_val
    if hasattr(model, 'dataset_factory'):
      if 'center' in model.dataset_factory.kwargs:
        assert center == model.dataset_factory.kwargs['center']

  def generate(self, x, **kwargs):
    if self.center:
      return -x
    return self.max_val - x


class Noise(Attack):
  """
  A weak attack that just picks a random point in the attacker's action space.
  When combined with an attack bundling function, this can be used to implement
  random search.

  References:
  https://arxiv.org/abs/1802.00420 recommends random search to help identify
    gradient masking.
  https://openreview.net/forum?id=H1g0piA9tQ recommends using noise as part
    of an attack bundling recipe combining many different optimizers to yield
    a stronger optimizer.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess=None, dtypestr='float32',
               **kwargs):

    super(Noise, self).__init__(model, sess=sess, dtypestr=dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'clip_min', 'clip_max')
    self.structural_kwargs = ['ord']

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    if self.ord != np.inf:
      raise NotImplementedError(self.ord)
    eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps,
                            dtype=self.tf_dtype)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      assert self.clip_min is not None and self.clip_max is not None
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    return adv_x

  def parse_params(self,
                   eps=0.3,
                   ord=np.inf,
                   clip_min=None,
                   clip_max=None,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:

    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # Save attack-specific parameters
    self.eps = eps
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf]:
      raise ValueError("Norm order must be np.inf")
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


class MaxConfidence(Attack):
  """
  The MaxConfidence attack.

  An attack designed for use against models that use confidence thresholding
  as a defense.
  If the underlying optimizer is optimal, this attack procedure gives the
  optimal failure rate for every confidence threshold t > 0.5.

  Publication: https://openreview.net/forum?id=H1g0piA9tQ

  :param model: cleverhans.model.Model
  :param sess: optional tf.session.Session
  :param base_attacker: cleverhans.attacks.Attack
  """

  def __init__(self, model, sess=None, base_attacker=None):
    if not isinstance(model, Model):
      raise TypeError("Model must be cleverhans.model.Model, got " +
                      str(type(model)))

    super(MaxConfidence, self).__init__(model, sess)
    if base_attacker is None:
      self.base_attacker = ProjectedGradientDescent(model, sess=sess)
    else:
      self.base_attacker = base_attacker
    self.structural_kwargs = self.base_attacker.structural_kwargs
    self.feedable_kwargs = self.base_attacker.feedable_kwargs

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    :param x: The model's symbolic inputs.
    :param kwargs: Keyword arguments for the base attacker
    """

    assert self.parse_params(**kwargs)
    labels, _nb_classes = self.get_or_guess_labels(x, kwargs)
    adv_x = self.attack(x, labels)

    return adv_x

  def parse_params(self, y=None, nb_classes=10, **kwargs):
    self.y = y
    self.nb_classes = nb_classes
    self.params = kwargs
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")
    return True

  def attack(self, x, true_y):
    adv_x_cls = []
    prob_cls = []
    m = tf.shape(x)[0]
    true_y_idx = tf.argmax(true_y, axis=1)

    expanded_x = tf.concat([x] * self.nb_classes, axis=0)
    target_ys = [tf.to_float(tf.one_hot(tf.ones(m, dtype=tf.int32) * cls,
                                        self.nb_classes))
                 for cls in range(self.nb_classes)]
    target_y = tf.concat(target_ys, axis=0)
    adv_x_cls = self.attack_class(expanded_x, target_y)
    expanded_all_probs = self.model.get_probs(adv_x_cls)

    adv_x_list = tf.split(adv_x_cls, self.nb_classes)
    all_probs_list = tf.split(expanded_all_probs, self.nb_classes)

    for cls in range(self.nb_classes):
      target_y = target_ys[cls]
      all_probs = all_probs_list[cls]
      # We don't actually care whether we hit the target class.
      # We care about the probability of the most likely wrong class
      cur_prob_cls = tf.reduce_max(all_probs - true_y, axis=1)
      # Knock out examples that are correctly classified.
      # This is not needed to be optimal for t >= 0.5, but may as well do it
      # to get better failure rate at lower thresholds.
      chosen_cls = tf.argmax(all_probs, axis=1)
      eligible = tf.to_float(tf.not_equal(true_y_idx, chosen_cls))
      cur_prob_cls = cur_prob_cls * eligible
      prob_cls.append(cur_prob_cls)

    probs = tf.concat([tf.expand_dims(e, 1) for e in prob_cls], axis=1)
    # Don't need to censor here because we knocked out the true class above
    # probs = probs - true_y
    most_confident = tf.argmax(probs, axis=1)
    fused_mask = tf.one_hot(most_confident, self.nb_classes)
    masks = tf.split(fused_mask, num_or_size_splits=self.nb_classes, axis=1)
    shape = [m] + [1] * (len(x.get_shape()) - 1)
    reshaped_masks = [tf.reshape(mask, shape) for mask in masks]
    out = sum(adv_x * rmask for adv_x,
              rmask in zip(adv_x_list, reshaped_masks))
    return out

  def attack_class(self, x, target_y):
    adv = self.base_attacker.generate(x, y_target=target_y, **self.params)
    return adv






def _project_perturbation(perturbation, epsilon, input_image, clip_min=None,
                          clip_max=None):
  """Project `perturbation` onto L-infinity ball of radius `epsilon`.
  Also project into hypercube such that the resulting adversarial example
  is between clip_min and clip_max, if applicable.
  """

  if clip_min is None or clip_max is None:
    raise NotImplementedError("_project_perturbation currently has clipping "
                              "hard-coded in.")

  # Ensure inputs are in the correct range
  with tf.control_dependencies([
      utils_tf.assert_less_equal(input_image,
                                 tf.cast(clip_max, input_image.dtype)),
      utils_tf.assert_greater_equal(input_image,
                                    tf.cast(clip_min, input_image.dtype))
  ]):
    clipped_perturbation = utils_tf.clip_by_value(
        perturbation, -epsilon, epsilon)
    new_image = utils_tf.clip_by_value(
        input_image + clipped_perturbation, clip_min, clip_max)
    return new_image - input_image


def projected_optimization(loss_fn,
                           input_image,
                           label,
                           epsilon,
                           num_steps,
                           clip_min=None,
                           clip_max=None,
                           optimizer=TensorAdam(),
                           project_perturbation=_project_perturbation,
                           early_stop_loss_threshold=None,
                           is_debug=False):
  """Generic projected optimization, generalized to work with approximate
  gradients. Used for e.g. the SPSA attack.

  Args:
    :param loss_fn: A callable which takes `input_image` and `label` as
                    arguments, and returns a batch of loss values. Same
                    interface as TensorOptimizer.
    :param input_image: Tensor, a batch of images
    :param label: Tensor, a batch of labels
    :param epsilon: float, the L-infinity norm of the maximum allowable
                    perturbation
    :param num_steps: int, the number of steps of gradient descent
    :param clip_min: float, minimum pixel value
    :param clip_max: float, maximum pixel value
    :param optimizer: A `TensorOptimizer` object
    :param project_perturbation: A function, which will be used to enforce
                                 some constraint. It should have the same
                                 signature as `_project_perturbation`.
    :param early_stop_loss_threshold: A float or None. If specified, the attack will end if the loss is below
       `early_stop_loss_threshold`.
        Enabling this option can have several different effects:
          - Setting the threshold to 0. guarantees that if a successful attack is found, it is returned.
            This increases the attack success rate, because without early stopping the optimizer can accidentally
            bounce back to a point where the attack fails.
          - Early stopping can make the attack run faster because it may run for fewer steps.
          - Early stopping can make the attack run slower because the loss must be calculated at each step.
            The loss is not calculated as part of the normal SPSA optimization procedure.
            For most reasonable choices of hyperparameters, early stopping makes the attack much faster because
            it decreases the number of steps dramatically.
    :param is_debug: A bool. If True, print debug info for attack progress.

  Returns:
    adversarial version of `input_image`, with L-infinity difference less than
      epsilon, which tries to minimize loss_fn.

  Note that this function is not intended as an Attack by itself. Rather, it
  is designed as a helper function which you can use to write your own attack
  methods. The method uses a tf.while_loop to optimize a loss function in
  a single sess.run() call.
  """
  assert num_steps is not None
  if is_debug:
    with tf.device("/cpu:0"):
      input_image = tf.Print(
          input_image, [],
          "Starting PGD attack with epsilon: %s" % epsilon)

  init_perturbation = tf.random_uniform(
      tf.shape(input_image),
      minval=tf.cast(-epsilon, input_image.dtype),
      maxval=tf.cast(epsilon, input_image.dtype),
      dtype=input_image.dtype)
  init_perturbation = project_perturbation(init_perturbation, epsilon,
                                           input_image, clip_min=clip_min,
                                           clip_max=clip_max)
  init_optim_state = optimizer.init_state([init_perturbation])
  nest = tf.contrib.framework.nest

  def loop_body(i, perturbation, flat_optim_state):
    """Update perturbation to input image."""
    optim_state = nest.pack_sequence_as(
        structure=init_optim_state, flat_sequence=flat_optim_state)

    def wrapped_loss_fn(x):
      return loss_fn(input_image + x, label)

    new_perturbation_list, new_optim_state = optimizer.minimize(
        wrapped_loss_fn, [perturbation], optim_state)
    projected_perturbation = project_perturbation(new_perturbation_list[0],
                                                  epsilon, input_image,
                                                  clip_min=clip_min,
                                                  clip_max=clip_max)

    # Be careful with this bool. A value of 0. is a valid threshold but evaluates to False, so we must explicitly
    # check whether the value is None.
    early_stop = early_stop_loss_threshold is not None
    compute_loss = is_debug or early_stop
    # Don't waste time building the loss graph if we're not going to use it
    if compute_loss:
      # NOTE: this step is not actually redundant with the optimizer step.
      # SPSA calculates the loss at randomly perturbed points but doesn't calculate the loss at the current point.
      loss = reduce_mean(wrapped_loss_fn(projected_perturbation), axis=0)

      if is_debug:
        with tf.device("/cpu:0"):
          loss = tf.Print(loss, [loss], "Total batch loss")

      if early_stop:
        i = tf.cond(tf.less(loss, early_stop_loss_threshold), lambda: float(num_steps), lambda: i)

    return i + 1, projected_perturbation, nest.flatten(new_optim_state)

  def cond(i, *_):
    return tf.less(i, num_steps)

  flat_init_optim_state = nest.flatten(init_optim_state)
  _, final_perturbation, _ = tf.while_loop(
      cond,
      loop_body,
      loop_vars=(tf.constant(0.), init_perturbation, flat_init_optim_state),
      parallel_iterations=1,
      back_prop=False,
      maximum_iterations=num_steps)
  if project_perturbation is _project_perturbation:
    # TODO: this assert looks totally wrong.
    # Not bothering to fix it now because it's only an assert.
    # 1) Multiplying by 1.1 gives a huge margin of error. This should probably
    #    take the difference and allow a tolerance of 1e-6 or something like
    #    that.
    # 2) I think it should probably check the *absolute value* of
    # final_perturbation
    perturbation_max = epsilon * 1.1
    check_diff = utils_tf.assert_less_equal(
        final_perturbation,
        tf.cast(perturbation_max, final_perturbation.dtype),
        message="final_perturbation must change no pixel by more than "
                "%s" % perturbation_max)
  else:
    # TODO: let caller pass in a check_diff function as well as
    # project_perturbation
    check_diff = tf.no_op()

  if clip_min is None or clip_max is None:
    raise NotImplementedError("This function only supports clipping for now")
  check_range = [utils_tf.assert_less_equal(input_image,
                                            tf.cast(clip_max,
                                                    input_image.dtype)),
                 utils_tf.assert_greater_equal(input_image,
                                               tf.cast(clip_min,
                                                       input_image.dtype))]

  with tf.control_dependencies([check_diff] + check_range):
    adversarial_image = input_image + final_perturbation
  return tf.stop_gradient(adversarial_image)
