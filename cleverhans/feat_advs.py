"""
This module inmplements a fast implementation of Feature Adversaries, an attack
against a target internal representation of a model.
Feature adversaries were originally introduced in (Sabour et al. 2016),
where the optimization was done using LBFGS.
Paper link: https://arxiv.org/abs/1511.05122
"""
import numpy as np
import tensorflow as tf

from cleverhans.attacks import Attack
from cleverhans.model import Model


def ffg(x, s_feat, g_feat, eps=0.3, ord=np.inf,
        clip_min=None, clip_max=None, targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    # feat.shape = (batch, c) or (batch, w, h, c)
    axis = range(1, len(s_feat.shape))

    # Compute loss (targeted)
    loss = tf.reduce_sum(tf.square(s_feat - g_feat), axis)

    if targeted:
        loss = - loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        signed_grad = tf.sign(grad)
    elif ord == 1:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.reduce_sum(tf.abs(grad),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True)
    elif ord == 2:
        reduc_ind = list(xrange(1, len(x.get_shape())))
        signed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                                   reduction_indices=reduc_ind,
                                                   keep_dims=True))
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


class FastFeatureGradient(Attack):

    """
    This is similar to FGSM but done for the internal representations.
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FastGradientMethod instance.
        """
        super(FastFeatureGradient, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'layer': str,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord']

    def generate(self, x, g, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        s_feat = self.model.get_layer(x, self.layer)
        g_feat = self.model.get_layer(g, self.layer)
        return ffg(x, s_feat, g_feat, eps=self.eps,
                   ord=self.ord, clip_min=self.clip_min,
                   clip_max=self.clip_max, targeted=self.targeted)

    def parse_params(self, layer=None, eps=0.3, ord=np.inf,
                     clip_min=None, clip_max=None, targeted=False, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.layer = layer
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th' and self.ord != np.inf:
            raise NotImplementedError("The only FastGradientMethod norm "
                                      "implemented for Theano is np.inf.")
        return True


class FastIterativeFeatureAdversaries(Attack):

    """
    This is similar to Basic Iterative Method (Kurakin et al. 2016) but
    applied to the internal representations.
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FeatureAdversaries instance.
        """
        super(FastIterativeFeatureAdversaries, self).__init__(model, back,
                                                              sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32,
                                'layer': str,
                                'targeted': False}
        self.structural_kwargs = ['ord', 'nb_iter']

        assert isinstance(self.model, Model)

    def generate(self, x, g, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param g: The target's symbolic representation.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        def clip_eta(eta):
            # Clipping perturbation eta to self.ord norm ball
            if self.ord == np.inf:
                eta = tf.clip_by_value(eta, -self.eps, self.eps)
            elif self.ord in [1, 2]:
                reduc_ind = list(xrange(1, len(eta.get_shape())))
                if self.ord == 1:
                    norm = tf.reduce_sum(tf.abs(eta),
                                         reduction_indices=reduc_ind,
                                         keep_dims=True)
                elif self.ord == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                                 reduction_indices=reduc_ind,
                                                 keep_dims=True))
                eta = eta * self.eps / norm
            return eta

        # Initialize loop variables
        eta = tf.truncated_normal(tf.shape(x), self.eps_iter / 2)
        eta = clip_eta(eta)

        fgm_params = {'eps': self.eps_iter, 'layer': self.layer,
                      'ord': self.ord, 'clip_min': self.clip_min,
                      'clip_max': self.clip_max, 'targeted': self.targeted}

        for i in range(self.nb_iter):
            FFG = FastFeatureGradient(self.model, back=self.back,
                                      sess=self.sess)
            # Compute this step's perturbation
            eta = FFG.generate(x + eta, g, **fgm_params) - x

            eta = clip_eta(eta)

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, layer=None, eps=0.3, eps_iter=0.05, nb_iter=10,
                     ord=np.inf, clip_min=None, clip_max=None,
                     targeted=False, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param layer: (required str) name of the layer to target.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
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
        self.targeted = targeted

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th':
            error_string = ("FeatureAdversaries is"
                            " not implemented in Theano")
            raise NotImplementedError(error_string)

        return True
