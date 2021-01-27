"""
The Attack class, providing a universal abstract interface describing attacks, and many implementations of it.
"""
from abc import ABCMeta
import collections
import warnings
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans import utils
from cleverhans.attacks.attack import Attack
from cleverhans.attacks.basic_iterative_method import BasicIterativeMethod
from cleverhans.attacks.carlini_wagner_l2 import CarliniWagnerL2
from cleverhans.attacks.deep_fool import DeepFool
from cleverhans.attacks.elastic_net_method import ElasticNetMethod
from cleverhans.attacks.fast_feature_adversaries import FastFeatureAdversaries
from cleverhans.attacks.fast_gradient_method import FastGradientMethod, fgm, optimize_linear
from cleverhans.attacks.lbfgs import LBFGS
from cleverhans.attacks.madry_et_al import MadryEtAl
from cleverhans.attacks.max_confidence import MaxConfidence
from cleverhans.attacks.momentum_iterative_method import MomentumIterativeMethod
from cleverhans.attacks.noise import Noise
from cleverhans.attacks.projected_gradient_descent import ProjectedGradientDescent
from cleverhans.attacks.saliency_map_method import SaliencyMapMethod
from cleverhans.attacks.semantic import Semantic
from cleverhans.attacks.spsa import SPSA, projected_optimization
from cleverhans.attacks.spatial_transformation_method import SpatialTransformationMethod
from cleverhans.attacks.virtual_adversarial_method import VirtualAdversarialMethod, vatm
from cleverhans.attacks.hop_skip_jump_attack import HopSkipJumpAttack, BoundaryAttackPlusPlus
from cleverhans.attacks.sparse_l1_descent import SparseL1Descent
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.model import wrapper_warning, wrapper_warning_logits
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans.utils_tf import clip_eta
from cleverhans import utils_tf

_logger = utils.create_logger("cleverhans.attacks")
tf_dtype = tf.as_dtype('float32')
