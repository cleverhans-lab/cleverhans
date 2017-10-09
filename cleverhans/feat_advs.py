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


class FastFeatureAdversaries(Attack):

    """
    This is similar to Basic Iterative Method (Kurakin et al. 2016) but
    applied to the internal representations.
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FastFeatureAdversaries instance.
        """
        super(FastFeatureAdversaries, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32,
                                'layer': str}
        self.structural_kwargs = ['ord', 'nb_iter']

        assert isinstance(self.model, Model)

    def parse_params(self, layer=None, eps=0.3, eps_iter=0.05, nb_iter=10,
                     ord=np.inf, clip_min=None, clip_max=None, **kwargs):
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

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th':
            error_string = ("FeatureAdversaries is"
                            " not implemented in Theano")
            raise NotImplementedError(error_string)

        return True

    def attack_single_step(self, x, eta, s_feat, g_feat):
        """
        TensorFlow implementation of the Fast Feature Gradient. This is a
        single step attack similar to Fast Gradient Method that attacks an
        internal representation.

        :param x: the input placeholder
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param s_feat: model's internal tensor for source
        :param g_feat: model's internal tensor for guide
        :return: a tensor for the adversarial example
        """
        from utils_tf import clip_eta

        adv_x = x + eta

        # feat.shape = (batch, c) or (batch, w, h, c)
        axis = range(1, len(s_feat.shape))

        # Compute loss
        loss = tf.reduce_sum(tf.square(s_feat - g_feat), axis)

        # Define gradient of loss wrt input
        grad, = tf.gradients(loss, x)

        # Multiply by constant epsilon
        scaled_signed_grad = self.eps_iter * tf.sign(grad)

        # Add perturbation to original example to obtain adversarial example
        adv_x = adv_x + scaled_signed_grad

        # If clipping is needed,
        # reset all values outside of [clip_min, clip_max]
        if (self.clip_min is not None) and (self.clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)

        return eta

    def generate(self, x, g, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param g: The target's symbolic representation.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from utils_tf import clip_eta

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        s_feat = self.model.get_layer(x, self.layer)
        g_feat = self.model.get_layer(g, self.layer)

        # Initialize loop variables
        eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
        eta = clip_eta(eta, self.ord, self.eps)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, s_feat, g_feat)

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x
