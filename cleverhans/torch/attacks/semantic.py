"""
Semantic adversarial Examples
"""

def semantic(x, center=True, max_val=1.):
  """
  Semantic adversarial examples.
  
  https://arxiv.org/abs/1703.06857

  Note: data must either be centered (so that the negative image can be
  made by simple negation) or must be in the interval of [-1, 1]

  Arguments
  ---------
  center : bool
           If true, assumes data has 0 mean so the negative image is just negation.
           If false, assumes data is in interval [0, max_val]
  max_val : float
            Maximum value allowed in the input data.
  """

  if self.center: return x*-1
  return self.max_val - x
