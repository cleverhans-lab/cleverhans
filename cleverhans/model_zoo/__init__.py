"""
A module for hosting a variety of models of interest to the adversarial
example community.

Warning:
  This module is not nearly as conservative as the rest of CleverHans.
  Most of CleverHans is used to create rigorous vulnerability benchmarks.
  For example, the error rate caused by an Attack is considered to be
  part of the API for that Attack, so we upgrade the major version number
  whenever it changes.
  Models in the model zoo can be tweaked regularly to improve accuracy,
  training speed, robustness, etc.
"""
