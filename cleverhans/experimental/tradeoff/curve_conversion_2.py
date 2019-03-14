from matplotlib import pyplot
import numpy as np
num_points = 10
r = np.array([0., 1e-3, .5])
a = np.array([1., .501, .5])
pyplot.subplot(1, 2, 1)
pyplot.plot([0, 1], [0, 1])
pyplot.plot(r, a)
alpha = np.linspace(0., 1., num_points)
pyplot.subplot(1, 2, 2)
for i in range(r.size):
  t = a[i] * np.ones_like(alpha) + alpha * (r[i] - a[i])
  pyplot.plot(alpha, t)
pyplot.show()
