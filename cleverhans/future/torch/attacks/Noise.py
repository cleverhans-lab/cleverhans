"""
The Noise Attack
"""

import numpy as np
import torch
import torch.nn as nn
from cleverhans.future.torch.attacks.Attack import Attack


class Noise(Attack):
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
        model: Model
        dtype: dtype of the data
        kwargs: passed through the super constructor
    """

    def __init__(self, model, dtype='float32', **kwargs):

        super(Noise, self).__init__(model, dtype=dtype, **kwargs)
        #self.feedable_kwargs = ('eps', 'clip_min', 'clip_max')
        #self.structural_kwargs = ['ord'] 

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        Args:
            x: The model's symbolic inputs.
            kwargs: See parse_parmas
        """

        assert self.parse_params(**kwargs)

        if self.ord != np.inf: raise NotImplementedError(self.ord)
        
        # TODO Check the dtype
        eta = torch.FloatTensor(*x.shape).uniform_(-self.eps, self.eps)

        adv_x = x + eta

        if self.clip_min is not None and self.clip_max is not None:
            adv_x = torch.clamp(adv_x, min=self.clip_min, max=self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, order=np.inf, clip_min=None, 
                     clip_max=None):
        """
        Take in a dictionary of parameters and applies attack-specific checks 
        before saving them as attributes.

        Attack-specific parameters:
            eps: (optional float) maximum distortion of the adversarial example
                  compared to original input
            ord: (optional) Order of the norm (mimics Numpy).
                 Possible values: np.inf
            clip_min: (optional float) Minimum input component value
            clip_max: (optional float) Maxiumum input component value
        """

        # Save attack specific parameters
        self.eps = eps
        self.ord = order
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf]: raise ValueError("Norm order must be in "
                                                      "np.inf")
        
        return True
