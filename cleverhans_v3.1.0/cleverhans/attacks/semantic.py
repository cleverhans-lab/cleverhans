"""Semantic adversarial examples
"""

from cleverhans.attacks.attack import Attack


class Semantic(Attack):
  """
  Semantic adversarial examples

  https://arxiv.org/abs/1703.06857

  Note: data must either be centered (so that the negative image can be
  made by simple negation) or must be in the interval [-1, 1]

  :param model: cleverhans.model.Model
  :param center: bool
    If True, assumes data has 0 mean so the negative image is just negation.
    If False, assumes data is in the interval [0, max_val]
  :param max_val: float
    Maximum value allowed in the input data
  :param sess: optional tf.Session
  :param dtypestr: dtype of data
  :param kwargs: passed through to the super constructor
  """

  def __init__(self, model, center, max_val=1., sess=None, dtypestr='float32',
               **kwargs):
    super(Semantic, self).__init__(model, sess, dtypestr, **kwargs)
    self.center = center
    self.max_val = max_val
    if hasattr(model, 'dataset_factory'):
      if 'center' in model.dataset_factory.kwargs:
        assert center == model.dataset_factory.kwargs['center']

  def generate(self, x, **kwargs):
    if self.center:
      return -x
    return self.max_val - x
