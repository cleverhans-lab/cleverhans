"""
Attacks for TensorFlow Eager
"""
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from cleverhans import attacks
from cleverhans import utils
from cleverhans.model import CallableModelWrapper, wrapper_warning
from cleverhans.model import Model
from cleverhans.loss import LossCrossEntropy

_logger = utils.create_logger("cleverhans.attacks_tfe")


if LooseVersion(tf.__version__) < LooseVersion('1.8.0'):
  error_msg = ('For eager execution',
               'use Tensorflow version greather than 1.8.0.')
  raise ValueError(error_msg)


class Attack(attacks.Attack):
  """
  Abstract base class for all eager attack classes.
  :param model: An instance of the cleverhans.model.Model class.
  :param back: The backend to use. Inherited from AttackBase class.
  :param dtypestr: datatype of the input data samples and crafted
                   adversarial attacks.
  """

  def __init__(self, model, dtypestr='float32'):
    super(Attack, self).__init__(model, dtypestr=dtypestr)
    # Validate the input arguments.
    if dtypestr != 'float32' and dtypestr != 'float64':
      raise ValueError("Unexpected input for argument dtypestr.")
    self.tf_dtype = tf.as_dtype(dtypestr)
    self.np_dtype = np.dtype(dtypestr)

    if not isinstance(model, Model):
      raise ValueError("The model argument should be an instance of"
                       " the cleverhans.model.Model class.")
    # Prepare attributes
    self.model = model
    self.dtypestr = dtypestr

  def construct_graph(self, **kwargs):
    """
    Constructs the graph required to run the attacks.
    Is inherited from the attack class, is overloaded
    to raise an error.
    """
    error = "This method is not required for eager execution."
    raise AttributeError(error)

  def generate_np(self, x_val, **kwargs):
    """
    Generate adversarial examples and return them as a NumPy array.

    :param x_val: A NumPy array with the original inputs.
    :param **kwargs: optional parameters used by child classes.
    :return: A NumPy array holding the adversarial examples.
    """
    tfe = tf.contrib.eager
    x = tfe.Variable(x_val)
    adv_x = self.generate(x, **kwargs)
    return adv_x.numpy()

  def construct_variables(self, kwargs):
    """
    Construct the inputs to the attack graph.
    Is inherited from the attack class, is overloaded
    to raise an error.
    """
    error = "This method is not required for eager execution."
    raise AttributeError(error)


class FastGradientMethod(Attack, attacks.FastGradientMethod):
  """
  Inherited class from Attack and cleverhans.attacks.FastGradientMethod.

  This attack was originally implemented by Goodfellow et al. (2015) with the
  infinity norm (and is known as the "Fast Gradient Sign Method"). This
  implementation extends the attack to other norms, and is therefore called
  the Fast Gradient Method.
  Paper link: https://arxiv.org/abs/1412.6572
  """

  def __init__(self, model, dtypestr='float32', **kwargs):
    """
    Creates a FastGradientMethod instance in eager execution.
    :model: cleverhans.model.Model
    :dtypestr: datatype in the string format.
    """
    del kwargs
    if not isinstance(model, Model):
      wrapper_warning()
      model = CallableModelWrapper(model, 'probs')

    super(FastGradientMethod, self).__init__(model, dtypestr)

  def generate(self, x, **kwargs):
    """
    Generates the adversarial sample for the given input.
    :param x: The model's inputs.
    :param eps: (optional float) attack step size (input variation)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param y: (optional) A tf variable` with the model labels. Only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param y_target: (optional) A tf variable` with the labels to target.
                        Leave y_target=None if y is also set.
                        Labels should be one-hot-encoded.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)
    labels, _nb_classes = self.get_or_guess_labels(x, kwargs)
    return self.fgm(x, labels=labels, targeted=(self.y_target is not None))

  def fgm(self, x, labels, targeted=False):
    """
    TensorFlow Eager implementation of the Fast Gradient Method.
    :param x: the input variable
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect.
                     Targeted will instead try to move in the direction
                     of being more like y.
    :return: a tensor for the adversarial example
    """
    # Compute loss
    with tf.GradientTape() as tape:
      # input should be watched because it may be
      # combination of trainable and non-trainable variables
      tape.watch(x)
      loss_obj = LossCrossEntropy(self.model, smoothing=0.)
      loss = loss_obj.fprop(x=x, y=labels)
      if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad = tape.gradient(loss, x)
    optimal_perturbation = attacks.optimize_linear(grad, self.eps, self.ord)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed
    # reset all values outside of [clip_min, clip_max]
    if (self.clip_min is not None) and (self.clip_max is not None):
      adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
    return adv_x


class BasicIterativeMethod(Attack, attacks.BasicIterativeMethod):
  """
  Inherited class from Attack and cleverhans.attacks.BasicIterativeMethod.

  The Basic Iterative Method (Kurakin et al. 2016). The original paper used
  hard labels for this attack; no label smoothing.
  Paper link: https://arxiv.org/pdf/1607.02533.pdf
  """

  FGM_CLASS = FastGradientMethod

  def __init__(self, model, dtypestr='float32'):
    """
    Creates a BasicIterativeMethod instance in eager execution.
    :param model: cleverhans.model.Model
    :param dtypestr: datatype in the string format.
    """
    if not isinstance(model, Model):
      wrapper_warning()
      model = CallableModelWrapper(model, 'probs')

    super(BasicIterativeMethod, self).__init__(model, dtypestr)
