"""
Data for possibilities frontier curves
"""

# SOTA results for CIFAR-10
# Threat model:
#  max norm constrained perturbation
#  eps = 8 when data in [0, 255]
# Note: the data for this frontier curve was made by running the
# compute_accuracy.py script. This is not a maximally aggressive
# evaluation.
# This data is useful for hyperparameter searches, comparing points
# in search space quickly to cheap baseline evaluations.
cifar10_max_norm_eps_8of255_compute_accuracy = [
  (.4547, .8725), # Madry et al 2017 adv_trained
  (.4523, .8714), # Madry et al 2017 secret
  (0., 0.9501), # Madry et al 2017 naturally trained
  (0., 0.9557) # cifar10_undefended.joblib
]
