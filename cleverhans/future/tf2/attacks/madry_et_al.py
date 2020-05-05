"""
The MadryEtAl attack
"""

from cleverhans.future.tf2.attacks.projected_gradient_descent import projected_gradient_descent


def madry_et_al(model_fn, x, eps, eps_iter, nb_iter, norm,
                clip_min=None, clip_max=None, y=None, targeted=False,
                rand_minmax=0.3, sanity_checks=True):
  """
  The attack from Madry et al 2017
  """
  return projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm,
                                    clip_min=clip_min, clip_max=clip_max, y=y, targeted=targeted,
                                    rand_init=True, rand_minmax=rand_minmax, sanity_checks=sanity_checks)
