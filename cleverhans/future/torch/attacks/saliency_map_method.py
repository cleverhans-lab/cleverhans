"""
The Saliency Map Method Attack
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from scratchai.attacks.attacks.attack import Attack

class SaliencyMapMethod(Attack):
  """
  The Jacobian-based Saliency Map Method (Papernot et al. 2016)
  Paper Link: https://arxiv.org/pdf/1511.07528.pdf

  Arguments
  ---------
  model : nn.Module
      The model on which the attack needs to be performed.
  dtype : str
      The data type of the model.

  Returns
  -------
  adv : torch.tensor
     The adversarial Example of the input.
  """

  def __init__(self, model, dtype='float32', **kwargs):
    super().__init__(model, dtype, **kwargs)

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.

    Arguments
    ---------
    x : torch.tensor
      The input to the model
    kwargs : dict
        Additonal arguments
    """

    # Parse and save attack specific parameters
    assert self.parse_params(**kwargs)

    if self.y_target is None:
      # TODO torch.autograd.grad doesn't support batches
      # So, revise the implementation when it does in future releases
      def random_targets(gt):
        result = gt.clone()
        classes = gt.shape[1]
        # TODO Remove the blank () after #18315 in pytorch
        return torch.roll(result, int(torch.randint(nb_classes, ())))
      
      labels, nb_classes = self.get_or_guess_labels(x, kwargs)
      self.y_target = random_targets(labels)
      self.y_target = self.y_target.view([1, nb_classes])
      #print (torch.argmax(self.y_target, dim=1))
    
    x_adv = jsma_symbolic(x, self.y_target, self.model, self.theta, 
                          self.gamma, self.clip_min, self.clip_max)
    return x_adv

  def parse_params(self, theta=1., gamma=1., clip_min=0., clip_max=1.,
          y_target=None):
    """
    Takes in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.

    Attack-specific parameters:
    ---------------------------
    theta : float, optional
        Perturbation introduced to modified components
        (can be positive or negative). Defaults to 1.
    gamma : float, optional
        Maximum percentage of perturbed features. Defaults to 1.
    clip_min : float, optional
         Minimum component value for clipping
    clip_max : float, optional
         Maximum component value for clipping
    y_target : torch.tensor, optional
         Target tensor if the attack is targetted
    """

    self.theta = theta
    self.gamma = gamma
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.y_target = y_target
    
    return True


def jsma_symbolic(x, y_target, net, theta, gamma, clip_min, clip_max):
  """
  PyTorch Implementation of the JSMA (see https://arxiv.org/abs/1511.07520
  for the details about the algorithm design choices).

  Arguments
  ---------
  x : torch.tensor
    The input to the model
  y_target : torch.tensor
       The target tensor
  model : nn.Module
      The pytorch model
  theta : float
      delta for each feature adjustment.
  gamma : float
      a float between 0 and 1 indicating the maximum distortion
      percentage.
  clip_min : float
       minimum value for components of the example returned
  clip_max : float
       maximum value for components of the example returned.
  
  Returns
  -------
  x_adv : torch.tensor
      The adversarial example.
  """

  classes = int(y_target.shape[1])
  features = int(np.product(x.shape[1:]).value)
  print (features)

  max_iters = np.floor(features * gamma / 2)
  increase = bool(theta > 0)

  zdiag = np.ones((features, features), int)
  np.fill_diagonal(zdiag, 0)

  # Compute the initial search domain. We optimize the initial search domain
  # by removing all features that are already at their maximum values
  # (if increasing input features -- otherwise, at their minimum value).
  if increase:
    search_domain = (x < clip_max).view(-1, features)
  else:
    search_domain = (x > clip_min).view(-1, features)
  
  print ('here')
  # TODO remove this
  max_iters = 1

  while max_iters:
    logits = net(x)
    preds  = torch.argmax(logits, dim=1)
  
    # Create the Jacobian Graph
    list_deriv = []
    for idx in range(classes):
      deriv = grad(logits[:, idx], x)
      print (deriv)
      list_deriv.append(deriv)

    grads = (torch.stack(list_deriv, dim=0).view(classes, -1, features))
    
    # Compute the Jacobian components
    # To help with the computation later, reshape the target_class
    # and other_class to [nb_classes, -1, 1].
    # The last dimension is added to allow broadcasting later.
    target_class = torch.view(classes, -1, 1)
    other_class = (target_class == 0).float()
    
    # TODO Check the dim
    grads_target = torch.sum(grads * target_class, dim=0)
    grads_other = torch.sum(grads * other_class, dim=0)
    
    # Remove the already-used input features from the search space
    # Subtract 2 times the maximum value from those so that they
    # won't be picked later
    increase_coef = (4 * int(increase) - 2) * (search_domain == 0).float()

    target_tmp = grads_target
    target_tmp -= increase_coef * torch.max(torch.abs(grads_target), dim=1)
    target_sum = target_tmp.view(-1, features, 1) + target_tmp.view(-1, 1, features)

    other_tmp = grads_other
    other_tmp -= increase_coef * torch.max(torch.abs(grads_target), dim=1)
    other_sum = other_tmp.view(-1, features, 1) * other_tmp.view(-1, 1, features)

    # Create a mask to only keep features that match conditions
    if increase:
      scores_mask = ((target_sum > 0) & (other_sum < 0))
    else:
      scores_mask = ((target_sum < 0) & (other_sum > 0))

    # Create a 2D numpy array of scores for each pair of candidate features
    scores = scores_mask * (-target_sum * other_sum) *  zdiag

    # Extract the best 2 pixels
    best = torch.argmax(scores.view(-1, features*features), dim=1)

    p1 = best % features
    p2 = best // features
    p1_ohot = one_hot(p1, depth=features)
    p2_ohot = one_hot(p2, depth=features)

    # Check if more modification is needed
    # TODO preds is 1 hot vector in tf implementation
    mod_not_done = torch.sum(y_target * preds, dim=1) == 0
    cond = mod_not_done & (torch.sum(search_domain, dim=1) >= 2)

    # Update the search domain
    cond_float = cond.view(-1, 1)
    to_mod = (p1_ohot + p2_ohot) * cond_float

    search_domain = search_domain - to_mod

    # Apply the modifications to the image
    to_mod = to_mod.view(-1, *x.shape)
    if increase:
      x = torch.min(clip_max, x + to_mod * theta)
    else:
      x = torch.min(clip_min, x + to_mod * theta)

    max_iters -= 1

  return x
