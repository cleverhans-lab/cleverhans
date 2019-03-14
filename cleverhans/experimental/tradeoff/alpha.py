"""
See alpha.lyx
"""
from matplotlib import pyplot
import numpy as np

def alpha_star(a0, r0, an, rn, test_hook=None):
  """
  Returns the alpha score defined in alpha.lyx

  :param a0: float, Accuracy of baseline model
  :param r0: float, Robustness (accuracy on advx) of baseline model
  :param an: float, Accuracy of new model
  :param rn: float, Robustness (accuracy on advx) of new model
  :param test_hook: dict, used for unit tests.
    Reports values of internal computations so tests can verify
    they hit all important cases.
  """
  if test_hook is None:
    test_hook = {}

  def valid_prob(p):
    if not isinstance(p, float):
      return False
    return p >= 0. and p <= 1.
  assert all(valid_prob(p) for p in [a0, r0, an, rn])

  c = an - a0 + r0 - rn
  test_hook['c'] = c
  numer = an - a0
  if c == 0:
    if a0 >= an:
      return 1.
    return 0.
  else:
    alpha_hat = numer / c
    test_hook['alpha_hat'] = alpha_hat
    if c < 0.:
      out = np.clip(alpha_hat, 0., 1.)
      # Don't return -0.
      if out == 0.:
        return 0.
      return out
    else:
      assert c > 0.
      if alpha_hat >= 1.:
        return 0.
      return 1.

def plot(a0, r0, an, rn, show=True):
  pyplot.plot([0, 1], [a0, r0], label="0")
  pyplot.plot([0, 1], [an, rn], label="n")
  pyplot.legend()
  if show:
    pyplot.show()

def case_0():
  # Negative c
  # negative alpha_hat

  a0 = .999
  an = 1.
  r0 = 0.
  rn = .1
  test_hook = {}
  score = alpha_star(a0=a0, an=an, r0=r0, rn=rn, test_hook=test_hook)
  assert test_hook['c'] < 0., test_hook['c']
  assert test_hook['alpha_hat'] < 0., test_hook['alpha_hat']
  # This plot should show that new model is always better
  # plot(a0, r0, an, rn)
  assert score == 0.

def case_1():
  # Negative c
  # alpha_hat in [0, 1]

  a0 = 1.
  an = .9
  r0 = 0.
  rn = .1
  test_hook = {}
  score = alpha_star(a0=a0, an=an, r0=r0, rn=rn, test_hook=test_hook)
  assert test_hook['c'] < 0., test_hook['c']
  alpha_hat = test_hook['alpha_hat']
  assert alpha_hat >= 0.
  assert alpha_hat <= 1.
  # This plot should show that the two curves intersect when
  # alpha = score and that the new model is better to the
  # right of that point
  # plot(a0, r0, an, rn)
  assert score == alpha_hat

def case_2():
  # Negative c
  # alpha_hat > 1

  a0 = 1.
  an = 0.
  r0 = 1.
  rn = 0.9
  test_hook = {}
  score = alpha_star(a0=a0, an=an, r0=r0, rn=rn, test_hook=test_hook)
  assert test_hook['c'] < 0., test_hook['c']
  alpha_hat = test_hook['alpha_hat']
  assert alpha_hat > 1., alpha_hat
  # This plot should show that the new model is always worse
  plot(a0, r0, an, rn)
  assert score == 1.

def case_3():
  # Zero c
  # a0 >= an

  a0 = 1.
  an = 0.
  r0 = 1.
  rn = 0.
  test_hook = {}
  score = alpha_star(a0=a0, an=an, r0=r0, rn=rn, test_hook=test_hook)
  assert test_hook['c'] == 0., test_hook['c']
  assert a0 >= an
  # This plot should show that the two lines are parallel and the
  # new model is always worse
  plot(a0, r0, an, rn)
  assert score == 1.

def case_4():
  # Zero c
  # a0 < an

  a0 = .9
  an = 1.
  r0 = 0.
  rn = .1
  test_hook = {}
  score = alpha_star(a0=a0, an=an, r0=r0, rn=rn, test_hook=test_hook)
  assert np.allclose(test_hook['c'], 0.), test_hook['c']
  assert a0 < an
  # This plot should show that the two lines are parallel and the
  # new model is always better
  plot(a0, r0, an, rn)
  assert score == 0.

def case_5():
  # Positive c
  # alpha_hat >= 1

  a0 = .9
  an = 1.
  r0 = 0.
  rn = 0.
  test_hook = {}
  score = alpha_star(a0=a0, an=an, r0=r0, rn=rn, test_hook=test_hook)
  assert test_hook['c'] > 0., test_hook['c']
  alpha_hat = test_hook['alpha_hat']
  # Enforce the generic definition of this case
  assert alpha_hat >= 1., alpha_hat
  # I want this test right on the boundary
  assert np.allclose(alpha_hat, 1.)
  # This plot should show that the new model is always better
  # Because the test is designed to put alpha_hat right on the boundary,
  # the two curves should touch at (1., 0.)
  plot(a0, r0, an, rn)
  assert score == 0.

def case_6():
  # Positive c
  # alpha_hat < 1

  a0 = .9
  an = 1.
  r0 = .1
  rn = 0.
  test_hook = {}
  score = alpha_star(a0=a0, an=an, r0=r0, rn=rn, test_hook=test_hook)
  assert test_hook['c'] > 0., test_hook['c']
  alpha_hat = test_hook['alpha_hat']
  assert alpha_hat < 1., alpha_hat
  # This plot should show that the new model is better but on the left
  # half of the plot which is not what alpha_score rewards
  plot(a0, r0, an, rn)
  assert score == 1.


if __name__ == "__main__":
  a0 = 1.
  r0 = 0.

  a0p = 1.
  r0p = .4

  an = .85
  rn = .45
  test_hook = {}

  anp = .94
  rnp = .4
  test_hook_p = {}

  print alpha_star(a0, r0, an, rn, test_hook)
  print alpha_star(a0, r0, anp, rnp, test_hook_p)
  print
  print alpha_star(a0p, r0p, an, rn, test_hook)
  print alpha_star(a0p, r0p, anp, rnp, test_hook_p)

  plot(a0, r0, an, rn, show=False)
  plot(a0, r0, anp, rnp)
