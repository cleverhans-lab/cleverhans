#!/usr/bin/env python3

import numpy as np
from cleverhans.attacks import FastGradientMethod
from robust_vision_benchmark import attack_server
from utils import cleverhans_attack_wrapper


def attack(model, session, a):
  fgsm = FastGradientMethod(model, sess=session)
  image = a.original_image[np.newaxis]
  return fgsm.generate_np(image)


attack_server(cleverhans_attack_wrapper(attack))
