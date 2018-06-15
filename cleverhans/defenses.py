from .loss import Loss
from .model import Model
import tensorflow as tf


class LossXEntropy(Loss):
    def __init__(self, model, smoothing, attack=None, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param smoothing: float, amount of label smoothing for cross-entropy.
        :param attack: Attack instance, the attack method for adv. training.
        """
        del kwargs
        Loss.__init__(self, model, locals(), attack)
        self.smoothing = smoothing

    def fprop(self, x, y, **kwargs):
        if self.attack is not None:
            x = x, self.attack(x)
        else:
            x = x,
        y -= self.smoothing * (y - 1. / tf.cast(y.shape[-1], tf.float32))
        logits = [self.model.get_logits(x, **kwargs) for x in x]
        loss = sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit)
            for logit in logits)
        return loss


class LossMixUp(Loss):
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
        mix = mix.sample([tf.shape(x)[0]] + [1]*(len(x.shape) - 1))
        xm = x + mix * (x[::-1] - x)
        ym = y + mix * (y[::-1] - y)
        logits = self.model.get_logits(xm, **kwargs)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=ym,
                                                          logits=logits)
        return loss


class LossFeaturePairing(Loss):
    def __init__(self, model, weight, attack, **kwargs):
        """Constructor.
        :param model: Model instance, the model on which to apply the loss.
        :param weight: float, with of logic pairing loss.
        :param attack: Attack instance, the attack method for adv. training.
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
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=d1[Model.O_LOGITS])
        loss += tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=d2[Model.O_LOGITS])
        return loss + self.weight * pairing_loss
