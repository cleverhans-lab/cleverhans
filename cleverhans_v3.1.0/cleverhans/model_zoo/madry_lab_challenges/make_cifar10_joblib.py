"""Makes a .joblib file containing the trained model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import logging

import tensorflow as tf
from tensorflow.python.platform import app, flags
from cleverhans.utils import set_log_level, to_categorical, safe_zip
from cleverhans.utils_tf import model_eval
from cleverhans import serial
from cleverhans.dataset import CIFAR10, Factory
from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet


FLAGS = flags.FLAGS


def main(argv):

  model_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  if model_file is None:
    print('No model found')
    sys.exit()

  set_log_level(logging.DEBUG)

  sess = tf.Session()
  with sess.as_default():

    model = make_wresnet()
    saver = tf.train.Saver()
    # Restore the checkpoint
    saver.restore(sess, model_file)
    SCOPE = "cifar10_challenge"
    model2 = make_wresnet(scope=SCOPE)
    assert len(model.get_vars()) == len(model2.get_vars())
    found = [False] * len(model2.get_vars())
    for var1 in model.get_vars():
      var1_found = False
      var2_name = SCOPE + "/" + var1.name
      for idx, var2 in enumerate(model2.get_vars()):
        if var2.name == var2_name:
          var1_found = True
          found[idx] = True
          sess.run(tf.assign(var2, var1))
          break
      assert var1_found, var1.name
    assert all(found)

    model2.dataset_factory = Factory(CIFAR10, {"max_val": 255})

    serial.save("model.joblib", model2)


if __name__ == '__main__':

  cifar10_root = os.environ['CIFAR10_CHALLENGE_DIR']
  default_ckpt_dir = os.path.join(cifar10_root, 'models/model_0')


  flags.DEFINE_string('checkpoint_dir', default_ckpt_dir,
                      'Checkpoint directory to load')

  app.run(main)
