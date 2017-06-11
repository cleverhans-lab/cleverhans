import numpy as np

import theano
import warnings
from theano import gradient, tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import utils_th

floatX = theano.config.floatX


def fgsm(x, predictions, eps, clip_min=None, clip_max=None):
    return fgm(x, predictions, y=None, eps=eps, ord=np.inf, clip_min=clip_min,
               clip_max=clip_max)


def fgm(x, predictions, y=None, eps=0.3, ord=np.inf, clip_min=None,
        clip_max=None):
    """
    Theano implementation of the Fast Gradient
    Sign method.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param y: the output placeholder. Use None (the default) to avoid the
            label leaking effect.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf (other norms not implemented yet).
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    warnings.warn("CleverHans support for Theano is deprecated and "
                  "will be dropped on 2017-11-08.")
    assert ord == np.inf, "Theano implementation not available for this norm."
    eps = np.asarray(eps, dtype=floatX)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = T.eq(predictions, T.max(predictions, axis=1, keepdims=True))
    y = T.cast(y, utils_th.floatX)
    y = y / T.sum(y, 1, keepdims=True)
    # Compute loss
    loss = utils_th.model_loss(y, predictions, mean=True)

    # Define gradient of loss wrt input
    grad = T.grad(loss, x)

    # Take sign of gradient
    signed_grad = T.sgn(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = gradient.disconnected_grad(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = T.clip(adv_x, clip_min, clip_max)

    return adv_x


def vatm(model, x, predictions, eps, num_iterations=1, xi=1e-6,
         clip_min=None, clip_max=None, seed=12345):
    """
    Theano implementation of the perturbation method used for virtual
    adversarial training: https://arxiv.org/abs/1507.00677
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param predictions: the model's unnormalized output tensor
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param seed: the seed for random generator
    :return: a tensor for the adversarial example
    """
    eps = np.asarray(eps, dtype=floatX)
    xi = np.asarray(xi, dtype=floatX)
    rng = RandomStreams(seed=seed)
    d = rng.normal(size=x.shape, dtype=x.dtype)
    for i in range(num_iterations):
        d = xi * utils_th.l2_batch_normalize(d)
        logits_d = model(x + d)
        kl = utils_th.kl_with_logits(predictions, logits_d)
        Hd = T.grad(kl.sum(), d)
        d = gradient.disconnected_grad(Hd)
    d = eps * utils_th.l2_batch_normalize(d)
    adv_x = gradient.disconnected_grad(x + d)
    if (clip_min is not None) and (clip_max is not None):
        adv_x = T.clip(adv_x, clip_min, clip_max)
    return adv_x
