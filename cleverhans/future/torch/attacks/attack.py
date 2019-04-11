"""
Base attack for PyTorch attacks.
"""

# TODO Deprecate. No need for a base class in the desired function-based API
class Attack:
  def __init__(self, model, dtype, **kwargs):
    self.model = model
    self.dtype = dtype  
