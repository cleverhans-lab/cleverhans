""" Boundary Attack++
"""
import logging
import numpy as np
import tensorflow as tf
from warnings import warn
from cleverhans.attacks import Attack
from cleverhans.model import CallableModelWrapper, Model, wrapper_warning_logits
from cleverhans import utils, utils_tf

np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.hop_skip_jump_attack")
_logger.setLevel(logging.INFO)


class HopSkipJumpAttack(Attack):
  """
  HopSkipJumpAttack was originally proposed by Chen, Jordan and Wainwright.
  It is a decision-based attack that requires access to output
  labels of a model alone.
  Paper link: https://arxiv.org/abs/1904.02144
  At a high level, this attack is an iterative attack composed of three
  steps: Binary search to approach the boundary; gradient estimation;
  stepsize search. HopSkipJumpAttack requires fewer model queries than
  Boundary Attack which was based on rejective sampling.
  :param model: cleverhans.model.Model
  :param sess: tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor.
  see parse_params for details.
  """

  def __init__(self, model, sess, dtypestr='float32', **kwargs):
    """
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """
    if not isinstance(model, Model):
      wrapper_warning_logits()
      model = CallableModelWrapper(model, 'logits')

    super(HopSkipJumpAttack, self).__init__(model, sess,
                                                 dtypestr, **kwargs)

    self.feedable_kwargs = ('y_target', 'image_target')

    self.structural_kwargs = [
        'stepsize_search',
        'clip_min',
        'clip_max',
        'constraint',
        'num_iterations',
        'initial_num_evals',
        'max_num_evals',
        'batch_size',
        'verbose',
        'gamma',
    ]

  def generate(self, x, **kwargs):
    """
    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.
    :param x: A tensor with the inputs.
    :param kwargs: See `parse_params`
    """
    self.parse_params(**kwargs)
    shape = [int(i) for i in x.get_shape().as_list()[1:]]

    assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
    _check_first_dimension(x, 'input')
    if self.y_target is not None:
      _check_first_dimension(self.y_target, 'y_target')
      assert self.image_target is not None, \
          'Require a target image for targeted attack.'
      _check_first_dimension(self.image_target, 'image_target')

    # Set shape and d.
    self.shape = shape
    self.d = int(np.prod(shape))

    # Set binary search threshold.
    if self.constraint == 'l2':
      self.theta = self.gamma / (np.sqrt(self.d) * self.d)
    else:
      self.theta = self.gamma / (self.d * self.d)

    # Construct input placeholder and output for decision function.
    self.input_ph = tf.placeholder(
        tf_dtype, [None] + list(self.shape), name='input_image')
    self.logits = self.model.get_logits(self.input_ph)

    def hsja_wrap(x, target_label, target_image):
      """ Wrapper to use tensors as input and output. """
      return np.array(self._hsja(x, target_label, target_image),
                      dtype=self.np_dtype)

    if self.y_target is not None:
      # targeted attack that requires target label and image.
      wrap = tf.py_func(hsja_wrap,
                        [x[0], self.y_target[0], self.image_target[0]],
                        self.tf_dtype)
    else:
      if self.image_target is not None:
        # untargeted attack with an initialized image.
        wrap = tf.py_func(lambda x, target_image: hsja_wrap(x,
                                                            None, target_image),
                          [x[0], self.image_target[0]],
                          self.tf_dtype)
      else:
        # untargeted attack without an initialized image.
        wrap = tf.py_func(lambda x: hsja_wrap(x, None, None),
                          [x[0]],
                          self.tf_dtype)

    wrap.set_shape(x.get_shape())

    return wrap

  def generate_np(self, x, **kwargs):
    """
    Generate adversarial images in a for loop.
    :param y: An array of shape (n, nb_classes) for true labels.
    :param y_target:  An array of shape (n, nb_classes) for target labels.
    Required for targeted attack.
    :param image_target: An array of shape (n, **image shape) for initial
    target images. Required for targeted attack.

    See parse_params for other kwargs.

    """

    x_adv = []

    if 'image_target' in kwargs and kwargs['image_target'] is not None:
      image_target = np.copy(kwargs['image_target'])
    else:
      image_target = None
    if 'y_target' in kwargs and kwargs['y_target'] is not None:
      y_target = np.copy(kwargs['y_target'])
    else:
      y_target = None

    for i, x_single in enumerate(x):
      img = np.expand_dims(x_single, axis=0)
      if image_target is not None:
        single_img_target = np.expand_dims(image_target[i], axis=0)
        kwargs['image_target'] = single_img_target
      if y_target is not None:
        single_y_target = np.expand_dims(y_target[i], axis=0)
        kwargs['y_target'] = single_y_target

      adv_img = super(HopSkipJumpAttack,
                      self).generate_np(img, **kwargs)
      x_adv.append(adv_img)

    return np.concatenate(x_adv, axis=0)

  def parse_params(self,
                   y_target=None,
                   image_target=None,
                   initial_num_evals=100,
                   max_num_evals=10000,
                   stepsize_search='geometric_progression',
                   num_iterations=64,
                   gamma=1.0,
                   constraint='l2',
                   batch_size=128,
                   verbose=True,
                   clip_min=0,
                   clip_max=1):
    """
    :param y: A tensor of shape (1, nb_classes) for true labels.
    :param y_target:  A tensor of shape (1, nb_classes) for target labels.
    Required for targeted attack.
    :param image_target: A tensor of shape (1, **image shape) for initial
    target images. Required for targeted attack.
    :param initial_num_evals: initial number of evaluations for
                              gradient estimation.
    :param max_num_evals: maximum number of evaluations for gradient estimation.
    :param stepsize_search: How to search for stepsize; choices are
                            'geometric_progression', 'grid_search'.
                            'geometric progression' initializes the stepsize
                             by ||x_t - x||_p / sqrt(iteration), and keep
                             decreasing by half until reaching the target
                             side of the boundary. 'grid_search' chooses the
                             optimal epsilon over a grid, in the scale of
                             ||x_t - x||_p.
    :param num_iterations: The number of iterations.
    :param gamma: The binary search threshold theta is gamma / d^{3/2} for
                   l2 attack and gamma / d^2 for linf attack.
    :param constraint: The distance to optimize; choices are 'l2', 'linf'.
    :param batch_size: batch_size for model prediction.
    :param verbose: (boolean) Whether distance at each step is printed.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    # ignore the y and y_target argument
    self.y_target = y_target
    self.image_target = image_target
    self.initial_num_evals = initial_num_evals
    self.max_num_evals = max_num_evals
    self.stepsize_search = stepsize_search
    self.num_iterations = num_iterations
    self.gamma = gamma
    self.constraint = constraint
    self.batch_size = batch_size
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.verbose = verbose

  def _hsja(self, sample, target_label, target_image):
    """
    Main algorithm for HopSkipJumpAttack.

    Return a tensor that constructs adversarial examples for the given
    input. Generate uses tf.py_func in order to operate over tensors.

    :param sample: input image. Without the batchsize dimension.
    :param target_label: integer for targeted attack,
      None for nontargeted attack. Without the batchsize dimension.
    :param target_image: an array with the same size as sample, or None.
      Without the batchsize dimension.


    Output:
    perturbed image.

    """

    # Original label required for untargeted attack.
    if target_label is None:
      original_label = np.argmax(
          self.sess.run(self.logits, feed_dict={self.input_ph: sample[None]})
          )
    else:
      target_label = np.argmax(target_label)

    def decision_function(images):
      """
      Decision function output 1 on the desired side of the boundary,
      0 otherwise.
      """
      images = clip_image(images, self.clip_min, self.clip_max)
      prob = []
      for i in range(0, len(images), self.batch_size):
        batch = images[i:i+self.batch_size]
        prob_i = self.sess.run(self.logits, feed_dict={self.input_ph: batch})
        prob.append(prob_i)
      prob = np.concatenate(prob, axis=0)
      if target_label is None:
        return np.argmax(prob, axis=1) != original_label
      else:
        return np.argmax(prob, axis=1) == target_label

    # Initialize.
    if target_image is None:
      perturbed = initialize(decision_function, sample, self.shape,
                             self.clip_min, self.clip_max)
    else:
      perturbed = target_image

    # Project the initialization to the boundary.
    perturbed, dist_post_update = binary_search_batch(sample,
                                                      np.expand_dims(perturbed, 0),
                                                      decision_function,
                                                      self.shape,
                                                      self.constraint,
                                                      self.theta)

    dist = compute_distance(perturbed, sample, self.constraint)

    for j in np.arange(self.num_iterations):
      current_iteration = j + 1

      # Choose delta.
      delta = select_delta(dist_post_update, current_iteration,
                           self.clip_max, self.clip_min, self.d,
                           self.theta, self.constraint)

      # Choose number of evaluations.
      num_evals = int(min([self.initial_num_evals * np.sqrt(j+1),
                           self.max_num_evals]))

      # approximate gradient.
      gradf = approximate_gradient(decision_function, perturbed, num_evals,
                                   delta, self.constraint, self.shape,
                                   self.clip_min, self.clip_max)
      if self.constraint == 'linf':
        update = np.sign(gradf)
      else:
        update = gradf

      # search step size.
      if self.stepsize_search == 'geometric_progression':
        # find step size.
        epsilon = geometric_progression_for_stepsize(perturbed,
                                                     update, dist, decision_function, current_iteration)

        # Update the sample.
        perturbed = clip_image(perturbed + epsilon * update,
                               self.clip_min, self.clip_max)

        # Binary search to return to the boundary.
        perturbed, dist_post_update = binary_search_batch(sample,
                                                          perturbed[None],
                                                          decision_function,
                                                          self.shape,
                                                          self.constraint,
                                                          self.theta)

      elif self.stepsize_search == 'grid_search':
        # Grid search for stepsize.
        epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
        epsilons_shape = [20] + len(self.shape) * [1]
        perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
        perturbeds = clip_image(perturbeds, self.clip_min, self.clip_max)
        idx_perturbed = decision_function(perturbeds)

        if np.sum(idx_perturbed) > 0:
          # Select the perturbation that yields the minimum distance # after binary search.
          perturbed, dist_post_update = binary_search_batch(sample,
                                                            perturbeds[idx_perturbed],
                                                            decision_function,
                                                            self.shape,
                                                            self.constraint,
                                                            self.theta)

      # compute new distance.
      dist = compute_distance(perturbed, sample, self.constraint)
      if self.verbose:
        print('iteration: {:d}, {:s} distance {:.4E}'.format(
            j+1, self.constraint, dist))

    perturbed = np.expand_dims(perturbed, 0)
    return perturbed


def BoundaryAttackPlusPlus(model, sess, dtypestr='float32', **kwargs):
  """
  A previous name used for HopSkipJumpAttack.
  """
  warn("BoundaryAttackPlusPlus will be removed after 2019-12-08; use HopSkipJumpAttack.")
  return HopSkipJumpAttack(model, sess, dtypestr, **kwargs)

def _check_first_dimension(x, tensor_name):
  message = "Tensor {} should have batch_size of 1.".format(tensor_name)
  if x.get_shape().as_list()[0] is None:
    check_batch = utils_tf.assert_equal(tf.shape(x)[0], 1, message=message)
    with tf.control_dependencies([check_batch]):
      x = tf.identity(x)
  elif x.get_shape().as_list()[0] != 1:
    raise ValueError(message)


def clip_image(image, clip_min, clip_max):
  """ Clip an image, or an image batch, with upper and lower threshold. """
  return np.minimum(np.maximum(clip_min, image), clip_max)


def compute_distance(x_ori, x_pert, constraint='l2'):
  """ Compute the distance between two images. """
  if constraint == 'l2':
    dist = np.linalg.norm(x_ori - x_pert)
  elif constraint == 'linf':
    dist = np.max(abs(x_ori - x_pert))
  return dist

def approximate_gradient(decision_function, sample, num_evals,
                         delta, constraint, shape, clip_min, clip_max):
  """ Gradient direction estimation """
  # Generate random vectors.
  noise_shape = [num_evals] + list(shape)
  if constraint == 'l2':
    rv = np.random.randn(*noise_shape)
  elif constraint == 'linf':
    rv = np.random.uniform(low=-1, high=1, size=noise_shape)

  axis = tuple(range(1, 1 + len(shape)))
  rv = rv / np.sqrt(np.sum(rv ** 2, axis=axis, keepdims=True))
  perturbed = sample + delta * rv
  perturbed = clip_image(perturbed, clip_min, clip_max)
  rv = (perturbed - sample) / delta

  # query the model.
  decisions = decision_function(perturbed)
  decision_shape = [len(decisions)] + [1] * len(shape)
  fval = 2 * decisions.astype(np_dtype).reshape(decision_shape) - 1.0

  # Baseline subtraction (when fval differs)
  if np.mean(fval) == 1.0:  # label changes.
    gradf = np.mean(rv, axis=0)
  elif np.mean(fval) == -1.0:  # label not change.
    gradf = - np.mean(rv, axis=0)
  else:
    fval = fval - np.mean(fval)
    gradf = np.mean(fval * rv, axis=0)

  # Get the gradient direction.
  gradf = gradf / np.linalg.norm(gradf)

  return gradf


def project(original_image, perturbed_images, alphas, shape, constraint):
  """ Projection onto given l2 / linf balls in a batch. """
  alphas_shape = [len(alphas)] + [1] * len(shape)
  alphas = alphas.reshape(alphas_shape)
  if constraint == 'l2':
    projected = (1-alphas) * original_image + alphas * perturbed_images
  elif constraint == 'linf':
    projected = clip_image(
        perturbed_images,
        original_image - alphas,
        original_image + alphas
    )
  return projected


def binary_search_batch(original_image, perturbed_images, decision_function,
                        shape, constraint, theta):
  """ Binary search to approach the boundary. """

  # Compute distance between each of perturbed image and original image.
  dists_post_update = np.array([
      compute_distance(
          original_image,
          perturbed_image,
          constraint
      )
      for perturbed_image in perturbed_images])

  # Choose upper thresholds in binary searchs based on constraint.
  if constraint == 'linf':
    highs = dists_post_update
    # Stopping criteria.
    thresholds = np.minimum(dists_post_update * theta, theta)
  else:
    highs = np.ones(len(perturbed_images))
    thresholds = theta

  lows = np.zeros(len(perturbed_images))

  while np.max((highs - lows) / thresholds) > 1:
    # projection to mids.
    mids = (highs + lows) / 2.0
    mid_images = project(original_image, perturbed_images,
                         mids, shape, constraint)

    # Update highs and lows based on model decisions.
    decisions = decision_function(mid_images)
    lows = np.where(decisions == 0, mids, lows)
    highs = np.where(decisions == 1, mids, highs)

  out_images = project(original_image, perturbed_images,
                       highs, shape, constraint)

  # Compute distance of the output image to select the best choice.
  # (only used when stepsize_search is grid_search.)
  dists = np.array([
      compute_distance(
          original_image,
          out_image,
          constraint
      )
      for out_image in out_images])
  idx = np.argmin(dists)

  dist = dists_post_update[idx]
  out_image = out_images[idx]
  return out_image, dist


def initialize(decision_function, sample, shape, clip_min, clip_max):
  """
  Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
  """
  success = 0
  num_evals = 0

  # Find a misclassified random noise.
  while True:
    random_noise = np.random.uniform(clip_min, clip_max, size=shape)
    success = decision_function(random_noise[None])[0]
    if success:
      break
    num_evals += 1
    message = "Initialization failed! Try to use a misclassified image as `target_image`"
    assert num_evals < 1e4, message

  # Binary search to minimize l2 distance to original image.
  low = 0.0
  high = 1.0
  while high - low > 0.001:
    mid = (high + low) / 2.0
    blended = (1 - mid) * sample + mid * random_noise
    success = decision_function(blended[None])[0]
    if success:
      high = mid
    else:
      low = mid

  initialization = (1 - high) * sample + high * random_noise
  return initialization


def geometric_progression_for_stepsize(x, update, dist, decision_function,
                                       current_iteration):
  """ Geometric progression to search for stepsize.
      Keep decreasing stepsize by half until reaching
      the desired side of the boundary.
  """
  epsilon = dist / np.sqrt(current_iteration)
  while True:
    updated = x + epsilon * update
    success = decision_function(updated[None])[0]
    if success:
      break
    else:
      epsilon = epsilon / 2.0

  return epsilon


def select_delta(dist_post_update, current_iteration,
                 clip_max, clip_min, d, theta, constraint):
  """
  Choose the delta at the scale of distance
   between x and perturbed sample.
  """
  if current_iteration == 1:
    delta = 0.1 * (clip_max - clip_min)
  else:
    if constraint == 'l2':
      delta = np.sqrt(d) * theta * dist_post_update
    elif constraint == 'linf':
      delta = d * theta * dist_post_update

  return delta
