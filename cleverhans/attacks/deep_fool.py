"""The DeepFool attack

"""
import warnings

import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.model import Model, wrapper_warning_logits, CallableModelWrapper


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
