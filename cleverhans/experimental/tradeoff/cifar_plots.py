from matplotlib import pyplot
import numpy as np

import hull

ran = [(.4551, .8786, "cifar10_adv_untargeted_pure_adv.joblib"),
       (0.4547, 0.8725, "madry_adv_trained.joblib"),
       (.4942, 0.8721, "cifar10_alp_1_epoch221.joblib (one-sided ALP)"),
       (.4912, .8839, "cifar10_adv_untargeted_pure_adv_smoothed_epoch224.joblib"),
       (.4759, .8728, "cifar10_alp_1.joblib"),
       (.4692, .8812, "cifar10_adv_untargeted_pure_adv_smoothed.joblib"),
       (0., .95, "fictional clean model"),
       (.455, .8809, "alp_0a1")
       ]

r = [e[0] for e in ran]
a = [e[1] for e in ran]
n = [e[2] for e in ran]


pyplot.subplot(1, 2, 1)
pyplot.scatter(r, a, alpha=0.5)
pyplot.xlabel('Adv accuracy')
pyplot.ylabel('Clean accuracy')

orig_points = zip(r, a)
points = hull.make_hull(orig_points)
print("Area below: ", hull.area_below(points))
r_hull, a_hull = zip(*points)
pyplot.scatter(r_hull, a_hull)


pyplot.subplot(1, 2, 2)
num_points = 2
alpha = np.linspace(0., 1., num_points)
for i in range(len(r)):
  t = a[i] * np.ones_like(alpha) + alpha * (r[i] - a[i])
  pyplot.plot(alpha, t)

pyplot.xlabel("Proportion adversarial")
pyplot.ylabel("Accuracy")

pyplot.show()
