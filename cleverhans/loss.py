import json
import os

from .model import Model
from .compat import softmax_cross_entropy_with_logits
import tensorflow as tf
import warnings

from cleverhans.utils import safe_zip


class Loss(object):
    """
    An abstract interface for loss wrappers that allows flexible control of
    real examples, adversarial examples and labels. These losses are used
    for defenses (during model training).
    """

    def __init__(self, model, hparams=None, attack=None):
        """
        :param model: Model instance, the model on which to apply the loss.
        :param hparams: dict, hyper-parameters for the loss.
        :param attack: callable, the attack function for adv. training.
        """
        assert isinstance(model, Model)
        assert attack is None or callable(attack)
        self.model = model
        self.hparams = hparams
        self.attack = attack

    def save(self, path):
        json.dump(dict(loss=self.__class__.__name__,
                       params=self.hparams),
                  open(os.path.join(path, 'loss.json'), 'wb'))

    def fprop(self, x, y):
        """Forward propagate the loss.
        Loss should be a scalar value, independent of batch size (i.e. use
        reduce_mean over batch axis, don't use reduce_sum or return a tensor).
        Scalar losses are easier to add together, e.g. through `WeightedSum`.
        Mean losses are easier to redistribute across multiple replicas without
        needing to change learning rates, etc.
        :param x: tensor, a batch of inputs.
        :param y: tensor, a batch of outputs (1-hot labels typically).
        """
        raise NotImplementedError


class WeightedSum(Loss):
    """
    A Loss that adds up a weighted sum of other losses.
    """
    def __init__(self, model, terms):
        self.terms = terms

        Loss.__init__(self, model, locals())

    def fprop(self, x, y, **kwargs):
        weights, loss_objects = safe_zip(*self.terms)
        for weight in weights:
            if isinstance(weight, float):
                continue
            if hasattr(weight, 'ndim'):
                assert weight.ndim == 0
                continue
            raise TypeError("weight of %s is not a type that this function "
                            "knows it can accept yet" % str(weight))
        losses = [loss.fprop(x, y, **kwargs) for loss in loss_objects]
        for loss, loss_object in safe_zip(losses, loss_objects):
            if len(loss.get_shape()) > 0:
                raise ValueError("%s.fprop returned a non-scalar value" %
                                 str(loss_object))
        terms = [weight * loss for weight, loss in safe_zip(weights, losses)]

        return tf.add_n(terms)


class CrossEntropy(Loss):
    def __init__(self, model, smoothing=0., attack=None, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param smoothing: float, amount of label smoothing for cross-entropy.
        :param attack: function, given an input x, return an attacked x'.
        """
        if smoothing < 0 or smoothing > 1:
            raise ValueError('Smoothing must be in [0, 1]', smoothing)
        self.kwargs = kwargs
        Loss.__init__(self, model, locals(), attack)
        self.smoothing = smoothing

    def fprop(self, x, y, **kwargs):
        kwargs.update(self.kwargs)
        if self.attack is not None:
            x = x, self.attack(x)
        else:
            x = x,

        # Catching RuntimeError: Variable -= value not supported by tf.eager.
        try:
            y -= self.smoothing * (y - 1. / tf.cast(y.shape[-1], y.dtype))
        except RuntimeError:
            y.assign_sub(self.smoothing * (y - 1. / tf.cast(y.shape[-1],
                                                            y.dtype)))

        logits = [self.model.get_logits(x, **kwargs) for x in x]
        loss = sum(
            tf.reduce_mean(softmax_cross_entropy_with_logits(labels=y,
                                                             logits=logit))
            for logit in logits)
        return loss


class MixUp(Loss):
    def __init__(self, model, beta, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param beta: float, beta distribution parameter for MixUp.
        """
        del kwargs
        Loss.__init__(self, model, locals())
        self.beta = beta

    def fprop(self, x, y, **kwargs):
        with tf.device('/CPU:0'):
            # Prevent error complaining GPU kernels unavailable for this.
            mix = tf.distributions.Beta(self.beta, self.beta)
            mix = mix.sample([tf.shape(x)[0]] + [1] * (len(x.shape) - 1))
        mix = tf.maximum(mix, 1 - mix)
        xm = x + mix * (x[::-1] - x)
        ym = y + mix * (y[::-1] - y)
        logits = self.model.get_logits(xm, **kwargs)
        loss = tf.reduce_mean(softmax_cross_entropy_with_logits(labels=ym,
                                                                logits=logits))
        return loss


class FeaturePairing(Loss):
    def __init__(self, model, weight, attack, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param weight: float, with of logic pairing loss.
        :param attack: function, given an input x, return an attacked x'.
        """
        del kwargs
        Loss.__init__(self, model, locals(), attack)
        self.weight = weight

    def fprop(self, x, y, **kwargs):
        x_adv = self.attack(x)
        d1 = self.model.fprop(x, **kwargs)
        d2 = self.model.fprop(x_adv, **kwargs)
        pairing_loss = [tf.reduce_mean(tf.square(a - b))
                        for a, b in
                        zip(d1[Model.O_FEATURES], d2[Model.O_FEATURES])]
        pairing_loss = tf.reduce_mean(pairing_loss)
        loss = tf.reduce_mean(softmax_cross_entropy_with_logits(
            labels=y, logits=d1[Model.O_LOGITS]))
        loss += tf.reduce_mean(softmax_cross_entropy_with_logits(
            labels=y, logits=d2[Model.O_LOGITS]))
        return loss + self.weight * pairing_loss


class WeightDecay(Loss):
    def fprop(self, x, y, **kwargs):
        terms = [tf.nn.l2_loss(param)
                 for param in self.model.get_params()
                 if len(param.get_shape()) > 1]
        out = tf.add_n(terms)
        assert len(out.get_shape()) == 0
        return out


def attack_softmax_cross_entropy(y, probs, mean=True):
    """
    Define target loss for an Attack.
    :param y: 2D tensor, one hot labels.
    :param probs: 2D tensor, probability distribution output from the model.
    :param mean: bool, reduce mean loss when true.
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """
    logits = probs.op.inputs[0] if probs.op.type == 'Softmax' else probs
    out = softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(out) if mean else out


class LossCrossEntropy(Loss):
    """
    Deprecated version of `CrossEntropy` that returns per-example loss rather
    than mean loss.
    """

    def __init__(self, model, smoothing=0., attack=None, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param smoothing: float, amount of label smoothing for cross-entropy.
        :param attack: function, given an input x, return an attacked x'.
        """
        if smoothing < 0 or smoothing > 1:
            raise ValueError('Smoothing must be in [0, 1]', smoothing)
        del kwargs
        Loss.__init__(self, model, locals(), attack)
        self.smoothing = smoothing

    def fprop(self, x, y, **kwargs):
        if self.attack is not None:
            x = x, self.attack(x)
        else:
            x = x,

        # Catching RuntimeError: Variable -= value not supported by tf.eager.
        try:
            y -= self.smoothing * (y - 1. / tf.cast(y.shape[-1], tf.float32))
        except RuntimeError:
            y.assign_sub(self.smoothing * (y - 1. / tf.cast(y.shape[-1],
                                                            tf.float32)))

        logits = [self.model.get_logits(x, **kwargs) for x in x]
        loss = sum(
            softmax_cross_entropy_with_logits(labels=y,
                                              logits=logit)
            for logit in logits)
        warnings.warn("LossCrossEntropy is deprecated, switch to "
                      "CrossEntropy. LossCrossEntropy may be removed on "
                      "or after 2019-03-06.")
        return loss


class LossFeaturePairing(Loss):
    """Deprecated version of `FeaturePairing` that returns per-example loss
    rather than mean loss."""

    def __init__(self, model, weight, attack, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param weight: float, with of logic pairing loss.
        :param attack: function, given an input x, return an attacked x'.
        """
        del kwargs
        Loss.__init__(self, model, locals(), attack)
        self.weight = weight

    def fprop(self, x, y, **kwargs):
        x_adv = self.attack(x)
        d1 = self.model.fprop(x, **kwargs)
        d2 = self.model.fprop(x_adv, **kwargs)
        pairing_loss = [tf.reduce_mean(tf.square(a - b))
                        for a, b in
                        zip(d1[Model.O_FEATURES], d2[Model.O_FEATURES])]
        pairing_loss = tf.reduce_mean(pairing_loss)
        loss = softmax_cross_entropy_with_logits(
            labels=y, logits=d1[Model.O_LOGITS])
        loss += softmax_cross_entropy_with_logits(
            labels=y, logits=d2[Model.O_LOGITS])
        warnings.warn("LossFeaturePairing is deprecated, switch to "
                      "FeaturePairing. LossFeaturePairing may be removed "
                      "on or after 2019-03-06.")
        return loss + self.weight * pairing_loss


class LossMixUp(Loss):
    """Deprecated version of `MixUp` that returns per-example loss
    rather than mean loss."""

    def __init__(self, model, beta, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param beta: float, beta distribution parameter for MixUp.
        """
        del kwargs
        Loss.__init__(self, model, locals())
        self.beta = beta

    def fprop(self, x, y, **kwargs):
        mix = tf.distributions.Beta(self.beta, self.beta)
        mix = mix.sample([tf.shape(x)[0]] + [1] * (len(x.shape) - 1))
        xm = x + mix * (x[::-1] - x)
        ym = y + mix * (y[::-1] - y)
        logits = self.model.get_logits(xm, **kwargs)
        loss = softmax_cross_entropy_with_logits(labels=ym, logits=logits)
        warnings.warn("LossMixUp is deprecated, switch to "
                      "MixUp. LossFeaturePairing may be removed "
                      "on or after 2019-03-06.")
        return loss
