"""
The MadryEtAl attack
"""

from cleverhans.attacks.projected_gradient_descent import ProjectedGradientDescent


class MadryEtAl(ProjectedGradientDescent):
  """
  The attack from Madry et al 2017
  """
  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    super(MadryEtAl, self).__init__(model, sess=sess,
                                    dtypestr=dtypestr,
                                    default_rand_init=True,
                                    **kwargs)
