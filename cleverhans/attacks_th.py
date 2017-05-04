import numpy as np
import theano
import theano.tensor as T
import warnings

from . import utils_th


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
    assert ord == np.inf, "Theano implementation not available for this norm."

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
    adv_x = theano.gradient.disconnected_grad(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = T.clip(adv_x, clip_min, clip_max)

    return adv_x
