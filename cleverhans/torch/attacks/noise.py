"""
The Noise Attack
"""

import numpy as np
import torch


def noise(x, eps=0.3, order=np.inf, clip_min=None, clip_max=None):
    """
    A weak attack that just picks a random point in the attacker's action
    space. When combined with an attack bundling function, this can be used to
    implement random search.

    References:
    https://arxiv.org/abs/1802.00420 recommends random search to help identify
        gradient masking

    https://openreview.net/forum?id=H1g0piA9tQ recommends using noise as part
        of an attack building recipe combining many different optimizers to
        yield a strong optimizer.

    Args:
    :param x: the input tensor
    :param eps: (optional float) maximum distortion of adversarial example 
                compared to original input.
    :param norm: (optional) Order of the norm.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """

    if order != np.inf: raise NotImplementedError(norm)
    
    eta = torch.FloatTensor(*x.shape).to(x.device).uniform_(-eps, eps)
    adv_x = x + eta

    if clip_min is not None or clip_max is not None:
        assert clip_min is not None and clip_max is not None
        adv_x = torch.clamp(adv_x, min=clip_min, max=clip_max)

    return adv_x
