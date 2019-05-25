"""
DeepFool Attack from https://arxiv.org/pdf/1511.04599.pdf 
by Seyed Mohsen et al.
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from torch.autograd.gradcheck import zero_gradients as zgrad


__all__ = ['deepfool']


def deepfool(x, net:nn.Module, eta:float=0.02, tnc:int=10, miter:int=50, 
             mret=False):
  """
  Implementation of the DeepFool Attack.

  Arguments
  ---------
  x : torch.Tensor
      The input tensor
  net : nn.Module
        The net.
  eta : float
        Eta as used in the paper.
  tnc : int
       The number of classes to consider.
  miter : int
          The maximum number of iterations.
  mret : bool
         If True returns the minimal perturbation that was added 
         to the original image. Defaults to False.
  """
  # Set net to eval
  net.eval()
  dev = x.device
  
  # x_idt is the initial advesarial image
  x_idt = x.detach().clone().requires_grad_(True)
  # Get the logits (since x_idt is not perturbed yet, the logits are true logits)
  logits = net(x_idt).squeeze()
  # Get the top tnc classes that the net is most confident about
  psort = logits.data.cpu().numpy().flatten().argsort()[::-1][:tnc]
  
  tlabl = psort[0] # True Label
  plabl = tlabl    # Pert label

  x_shape = x_idt.squeeze().shape # [C x H x W]
  w = np.zeros(x_shape)           # [C x H x W]
  rt = np.zeros(x_shape)          # [C x H x W]
  
  i = 0
  while plabl == tlabl and i < miter:
    # Initial Perturbation
    pert = np.inf
    logits[tlabl].backward(retain_graph=True)
    ograd = x_idt.grad.data.cpu().numpy().copy()

    for c in range(1, tnc):
      zgrad(x_idt)
      logits[psort[c]].backward(retain_graph=True)
      cgrad = x_idt.grad.data.cpu().numpy().copy()

      # Get new wk and fk
      wk = cgrad - ograd
      fk = (logits[c] - logits[tlabl]).item()

      cpert = abs(fk) / np.linalg.norm(wk.flatten())
      if cpert < pert: pert = cpert; w = wk
    
    # Added 1e-4 for numerical stability
    ri =  (pert+1e-4) * w / np.linalg.norm(w.flatten())
    rt += ri.squeeze()
    
    x_idt = x + ((1+eta) * torch.from_numpy(rt).unsqueeze(0)).float().to(dev)
    logits = net(x_idt.requires_grad_(True)).squeeze()
    plabl = torch.argmax(logits, dim=0).item()

    i += 1
  
  rt = (1+eta) * rt
  return (x_idt, rt) if mret else x_idt
