import json
import os
import tensorflow as tf

from .model import Model


class Loss(object):
    """
    An abstract interface for loss wrappers that allows flexible control of
    real examples, adversarial examples and labels..
    """

    def __init__(self, model, hparams, attack=None):
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
        :param x: tensor, a batch of inputs.
        :param y: tensor, a batch of outputs (1-hot labels typically).
        """
        raise NotImplementedError


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
    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return tf.reduce_mean(out) if mean else out
