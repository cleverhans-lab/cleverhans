import six
import math
import time
import os
import random
import logging

import tensorflow as tf

from cleverhans.utils_tf import batch_indices

from model import clone_variable

from evaluator import create_adv_by_name
from manager import TrainManager


class TrainerMultiGPU(TrainManager):
    def __init__(self, **kwargs):
        super(TrainerMultiGPU, self).__init__(**kwargs)

