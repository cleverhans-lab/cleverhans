"""
This tutorial shows how to train a deflecting model based on CapsLayer with Tensorflow.
The original paper can be found at:

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import sys
sys.path.append("/home/yaoqin/cleverhans/")
from cleverhans.augmentation import random_shift, random_horizontal_flip
from cleverhans.compat import flags
from cleverhans.dataset import SVHN, CIFAR10
from cleverhans.loss import MarginCycLoss
from cleverhans.model_zoo.capsnet_deflect import CapsNetRecons
from cleverhans.train import train
from cleverhans.utils_tf import model_eval
from cleverhans_tutorials import check_installation

FLAGS = flags.FLAGS
DATASET = 'SVHN'
if DATASET == 'SVHN':
  BATCH_SIZE = 64
  IMAGE_SIZE = 64000000
  NB_EPOCHS = int(IMAGE_SIZE/50000.)
  NUM_CAPSULES_OUTPUT = 25
  OUTPUT_ATOMS = 4
  NUM_ROUTING = 1
  LEARNING_RATE = 0.0001
  NB_FILTERS = 64
  TRAIN_END = 73257
  TEST_END = 26032
elif DATASET == 'CIFAR10':
  BATCH_SIZE = 128
  IMAGE_SIZE = 64000000
  NB_EPOCHS = int(IMAGE_SIZE/50000.)
  NUM_CAPSULES_OUTPUT = 25
  OUTPUT_ATOMS = 8
  NUM_ROUTING = 1
  LEARNING_RATE = 0.0002
  NB_FILTERS = 128
  TRAIN_END = 60000
  TEST_END = 10000
else:
  print("Only SVHN and CIFAR10 are supported!!")


def train_deflecting(dataset_name=DATASET, train_start=0, train_end=TRAIN_END, test_start=0,
                     test_end=TEST_END, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                     num_capsules_output=NUM_CAPSULES_OUTPUT,
                     output_atoms = OUTPUT_ATOMS,
                     num_routing = NUM_ROUTING,
                     learning_rate=LEARNING_RATE,
                     nb_filters=NB_FILTERS, num_threads=None):
  """
  SVHN cleverhans tutorial to train a deflecting model based on CapsLayer
  :dataset_name: SVHN or CIFAR10
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param num_capsules_output: number of output capsules
  :param output_atoms: size of each capsule vector
  :param num_routing: number of routings in capsule layer
  :param learning_rate: learning rate for training

  """

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  # Get svhn data
  if dataset_name == "SVHN": 
    data = SVHN(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  elif dataset_name == "CIFAR10":
    data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  
  if dataset_name == "SVHN": 
    dataset_train = dataset_train.map(lambda x, y: (random_shift((x)), y), 4)
  elif dataset_name == "CIFAR10":
    dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')  
  x_test, y_test = data.get_set('test')
  
  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))


  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  eval_params = {'batch_size': batch_size}
  
  rng = np.random.RandomState([2017, 8, 30])

    
  model = CapsNetRecons(dataset_name, nb_classes, nb_filters, input_shape=[batch_size, img_rows, img_cols, nchannels], num_capsules_output=num_capsules_output, output_atoms=output_atoms, num_routing=num_routing)
  var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dataset_name)
  
  preds = model.get_logits(x)   
  loss = MarginCycLoss(model)

  def evaluate():
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params) 
    print('Test accuracy on %s examples: %0.4f' % ("clean", acc))
    return acc
 
  train(sess, loss, None, None,
        dataset_train=dataset_train, dataset_size=dataset_size,
        evaluate=evaluate, args=train_params, rng=rng,
        var_list=var_lists)  


def main(argv=None):
  
  check_installation(__file__)

  train_deflecting(dataset_name=FLAGS.dataset, 
                   train_end=FLAGS.train_end, 
                   test_end=FLAGS.test_end, 
                   nb_epochs=FLAGS.nb_epochs, 
                   batch_size=FLAGS.batch_size, 
                   num_capsules_output=FLAGS.num_capsules_output, 
                   output_atoms=FLAGS.output_atoms,
                   num_routing=FLAGS.num_routing,
                   learning_rate=FLAGS.learning_rate,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':  
  flags.DEFINE_integer('train_end', TRAIN_END,
                       'Number of training data')
  flags.DEFINE_integer('test_end', TEST_END,
                       'Number of test data')
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_integer('num_capsules_output', NUM_CAPSULES_OUTPUT,
                       'Number of class capsules and background capsules')
  flags.DEFINE_integer('output_atoms', OUTPUT_ATOMS,
                       'Size of each capsule')
  flags.DEFINE_integer('num_routing', NUM_ROUTING,
                       'Number of routing in capsule layer')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('dataset', DATASET, 'SVHN or CIFAR10')

  tf.app.run()
