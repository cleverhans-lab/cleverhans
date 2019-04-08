"""
Base attack for PyTorch attacks.
"""
import torch

class Attack:
  def __init__(self, model, dtype, **kwargs):
    self.model = model
    self.dtype = dtype

  def get_or_guess_labels(self, x, kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' is in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.
    """
    if 'y' in kwargs and 'y_target' in kwargs:
      raise ValueError("Can not set both 'y' and 'y_target'.")
    if 'y' in kwargs:
      labels = kwargs['y']
    elif 'y_target' in kwargs and kwargs['y_target'] is not None:
      labels = kwargs['y_target']
    else:
      _, labels = torch.max(self.model(x), 1)
    return labels
