"""
The BasicIterativeMethod attack.
"""

from cleverhans.attacks.projected_gradient_descent import ProjectedGradientDescent


class BasicIterativeMethod(ProjectedGradientDescent):
  """
  The BasicIterativeMethod attack.
  """
  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(BasicIterativeMethod, self).__init__(model, sess=sess,
                                               dtypestr=dtypestr,
                                               default_rand_init=False,
                                               **kwargs)
