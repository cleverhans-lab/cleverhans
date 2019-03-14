"""
Coverage project model for CIFAR
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import numpy as np
import os
import tensorflow as tf
from tensorflow.python.platform import flags
import time

from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.dataset import CIFAR10
from cleverhans.evaluation import accuracy
from cleverhans.loss import CrossEntropy, MixUp
from cleverhans.utils_tf import model_eval
from cleverhans.augmentation import random_crop_and_flip, \
    random_horizontal_flip, random_shift, batch_augment
from cleverhans.train import train
from cleverhans.utils_tf import infer_devices
import cifar10_model
from cleverhans.serial import save
from loss import LossCrossEntropyInstanceNoise, WeightedSum, WeightDecay
import compute_accuracy
from frontiers import cifar10_max_norm_eps_8of255_compute_accuracy as orig_points
import hull

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

epoch = 0
best_epoch = -1
best_area = 0.
tiebreak_clean_acc = 0.
last_test_print = time.time()
last_train_print = time.time()

assert __file__.startswith('train_')
assert __file__.endswith('.py')
SAVE_PATH = __file__[len('train_'):-len(".py")] + ".joblib"
MODEL_NAME = "Model_dropout"
TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
NB_EPOCHS = 350
devices = infer_devices()
num_devices = len(devices)
BATCH_SIZE = 64 * num_devices
GCN = False
LEARNING_RATE = 5e-4
LOGIT_SCALE = 2.
LOGIT_NORM = 1  # These are really bools but I don't like the FLAGS interface for bools
ORIG_LOGIT_NORM = 0
USE_EMA = 1
EMA_DECAY = 'ema_decay_orig'
WEIGHT_DECAY = 0.
CLEAN_LABEL_SMOOTHING = 0.0
NB_FILTERS = 256
USE_INPUT_DROPOUT = 0
HID_DROPOUT_KEEP_PROB = 1.
SHIFT_MODE = 'REFLECT'
MIXUP = 0.5


def ema_decay_orig(epoch, batch):

  def return_0(): return 0.

  def return_999(): return .999

  def return_9999(): return .9999

  def inner_cond(): return tf.cond(epoch < 100, return_999, return_9999)

  out = tf.cond(epoch < 10, return_0, inner_cond)
  return out


def ema_decay_2(epoch, batch):
  def return_0(): return 0.

  def return_9999(): return .9999

  out = tf.cond(epoch < 30, return_0, return_9999)
  return out


def do_train(train_start=TRAIN_START, train_end=60000, test_start=0,
             test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
             learning_rate=LEARNING_RATE,
             backprop_through_attack=False,
             nb_filters=NB_FILTERS, num_threads=None,
             logit_scale=LOGIT_SCALE,
             gcn=GCN,
             logit_norm=LOGIT_NORM,
             orig_logit_norm=ORIG_LOGIT_NORM,
             use_ema=USE_EMA,
             ema_decay=EMA_DECAY,
             weight_decay=WEIGHT_DECAY,
             use_input_dropout=USE_INPUT_DROPOUT,
             model_name=MODEL_NAME,
             hid_dropout_keep_prob=HID_DROPOUT_KEEP_PROB,
             clean_label_smoothing=CLEAN_LABEL_SMOOTHING,
             noisy_label_smoothing=0.4,
             shift_mode=SHIFT_MODE,
             mixup=MIXUP):
  """
  Train coverage project CIFAR-10 model
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  """
  print('Parameters')
  print('-'*79)
  for x, y in sorted(locals().items()):
    print('%-32s %s' % (x, y))
  print('-'*79)

  if os.path.exists(FLAGS.save_path):
    print("Model " + FLAGS.save_path + " already exists. Refusing to overwrite.")
    quit()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  dataset = CIFAR10(train_start=train_start, train_end=train_end,
                    test_start=test_start, test_end=test_end,
                    center=True)
  assert dataset.x_train.shape[0] == 50000
  x_subtrain = dataset.x_train[:40000]
  y_subtrain = dataset.y_train[:40000]
  x_valid = dataset.x_train[40000:]
  assert x_valid.shape[0] == 10000
  y_valid = dataset.y_train[40000:]

  # Use Image Parameters
  img_rows, img_cols, nchannels = dataset.x_train.shape[1:4]
  nb_classes = dataset.NB_CLASSES

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  train_params = {
      'nb_epochs': nb_epochs,
      'learning_rate': learning_rate,
      'batch_size': batch_size,
  }
  eval_params = {'batch_size': batch_size}
  rng = np.random.RandomState([2017, 8, 30])
  sess = tf.Session()

  def do_eval(x_set, y_set, is_adv=None):
    acc = accuracy(sess, model, x_set, y_set)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'clean'
    if report_text:
      print('Accuracy on %s examples: %0.4f' % (report_text, acc))
    return acc

  model_class = getattr(cifar10_model, model_name)
  model = model_class(logit_scale=logit_scale, gcn=gcn,
                      filters=nb_filters,
                      logit_norm=logit_norm, orig_logit_norm=orig_logit_norm)
  factory = dataset.get_factory()
  model.dataset_factory = factory


  pgd = ProjectedGradientDescent(model=model, sess=sess)

  center = dataset.kwargs['center']
  value_range = 1. + center
  base_eps = 8. / 255.

  attack_params = {
    'eps' : base_eps * value_range,
    'clip_min' : -float(center),
    'clip_max' : float(center),
    'eps_iter' : (2. / 255.) * value_range,
    'nb_iter' : 40.
  }

  primary_loss = CrossEntropy(
        model, smoothing=clean_label_smoothing,
        attack=pgd,
        adv_coeff=1.,
        attack_params=attack_params,
        cifar10_model_hid_keep_prob=hid_dropout_keep_prob
    )

  if weight_decay > 0.:
    weight_decay_loss = WeightDecay(model)
    loss = WeightedSum(
        model, [(1., primary_loss), (weight_decay, weight_decay_loss)])
  else:
    loss = primary_loss

  print_test_period = 10
  print_train_period = 50

  def evaluate():
    global epoch
    global last_test_print
    global last_train_print
    global best_area
    global tiebreak_clean_acc
    global best_epoch
    with sess.as_default():
        print("Saving to ", FLAGS.save_path)
        save(FLAGS.save_path, model)
    if epoch % print_test_period == 0 or time.time() - last_test_print > 300:
      t1 = time.time()
      result = compute_accuracy.impl(sess, model, dataset, factory, x_valid, y_valid)
      t2 = time.time()
      print("Result.keys(): ", result.keys())
      new_point = (result['pgd'], result['clean'])
      points = orig_points + [new_point]
      area = hull.area_below(hull.make_hull(points))
      print("Area under hull: ", area)
      if area > best_area:
        best_epoch = epoch
        best_area = area
        tiebreak_clean_acc = new_point[1]
        print("Best area so far")
      elif area == best_area:
        if new_point[1] > tiebreak_clean_acc:
          tiebreak_clean_acc = new_point[1]
          best_epoch = epoch
          print("Tied for best area, wins on clean accuracy")
      print("Best area so far: ", best_area)
      print("Best clean accuracy for that area: ", tiebreak_clean_acc)
      print("Best epoch: ", best_epoch)
      last_test_print = t2
      print("Test eval time: ", t2 - t1)
    if epoch % print_train_period == 0 or time.time() - last_train_print > 3000:
      t1 = time.time()
      print("Training set: ")
      do_eval(x_subtrain, y_subtrain, False)
      t2 = time.time()
      print("Train eval time: ", t2 - t1)
      last_train_print = t2
    epoch += 1

  # optimizer = tf.train.MomentumOptimizer(.1 * batch_size / 128, .9, use_nesterov=False)
  optimizer = None

  def augmenter(x):
    def func(x):
      return random_horizontal_flip(random_shift(x, mode=shift_mode))

    return batch_augment(x, func)

  if shift_mode == 'REFERENCE':
    augmenter = random_crop_and_flip
  if use_input_dropout:
    def dropout(_x):
      _x1, _x2 = tf.split(_x, 2)
      # Use dropout on only half of the training examples
      # Dropout on all examples was hurting performance
      _x1 = tf.nn.dropout(_x1, keep_prob=.8)
      return tf.concat((_x1, _x2), axis=0)

    def compose(f, g):
      class Composition(object):
        def __init__(self, f, g):
          self.f = f
          self.g = g

        def __call__(self, _x):
          return self.f(self.g(_x))
      return Composition(f, g)
    augmenter = compose(augmenter, dropout)

  ema_decay = globals()[ema_decay]
  assert callable(ema_decay)

  train(sess, loss, x_subtrain, y_subtrain, evaluate=evaluate,
        optimizer=optimizer,
        args=train_params, rng=rng, var_list=model.get_params(),
        use_ema=use_ema, ema_decay=ema_decay,
        x_batch_preprocessor=augmenter)
  # Make sure we always evaluate on the last epoch, so pickling bugs are more
  # obvious
  if (epoch - 1) % print_test_period != 0:
    do_eval(dataset.x_test, dataset.y_test, False)
  if (epoch - 1) % print_train_period != 0:
    print("Training set: ")
    do_eval(dataset.x_train, dataset.y_train, False)

  with sess.as_default():
    save(FLAGS.save_path, model)
    # Now that the model has been saved, you can evaluate it in a
    # separate process using `evaluate_pickled_model.py`.
    # You should get exactly the same result for both clean and
    # adversarial accuracy as you get within this program.


def main(argv=None):
  name_of_script = argv[0]
  if len(argv) > 1:
    raise ValueError("Unparsed arguments to script: ", argv[1:])
  do_train(clean_label_smoothing=FLAGS.clean_label_smoothing,
           train_end=FLAGS.train_end, test_end=FLAGS.test_end,
           noisy_label_smoothing=FLAGS.noisy_label_smoothing,
           nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
           learning_rate=FLAGS.learning_rate,
           backprop_through_attack=FLAGS.backprop_through_attack,
           gcn=FLAGS.gcn,
           logit_scale=FLAGS.logit_scale,
           logit_norm=FLAGS.logit_norm,
           orig_logit_norm=FLAGS.orig_logit_norm,
           use_ema=FLAGS.use_ema,
           ema_decay=FLAGS.ema_decay,
           weight_decay=FLAGS.weight_decay,
           use_input_dropout=FLAGS.use_input_dropout,
           hid_dropout_keep_prob=FLAGS.hid_dropout_keep_prob,
           model_name=FLAGS.model_name,
           nb_filters=FLAGS.nb_filters,
           shift_mode=FLAGS.shift_mode,
           mixup=FLAGS.mixup)


if __name__ == '__main__':
  flags.DEFINE_string('save_path', SAVE_PATH, 'Path to save to')
  flags.DEFINE_integer('train_end', TRAIN_END,
                       'Ending index of range of training examples to use')
  flags.DEFINE_integer('test_end', TRAIN_END,
                       'Ending index of range of test examples to use')
  flags.DEFINE_integer('nb_filters', NB_FILTERS, 'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
  flags.DEFINE_integer('logit_norm', LOGIT_NORM, 'Whether to use logit norm')
  flags.DEFINE_integer('orig_logit_norm', ORIG_LOGIT_NORM,
                       'Whether to use original version of logit norm')
  flags.DEFINE_string('shift_mode', SHIFT_MODE,
                      'Shift mode for augmentation {REFLECT, CONSTANT, SYMMETRIC}')
  flags.DEFINE_float('mixup', MIXUP, 'Mixup coefficient [0, +inf[')
  flags.DEFINE_integer('use_ema', USE_EMA, 'Whether to use EMA')
  flags.DEFINE_string('ema_decay', EMA_DECAY,
                      'Name of function to use for EMA decay schedule')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_float('weight_decay', WEIGHT_DECAY, 'Weight decay coefficient')
  flags.DEFINE_float('logit_scale', LOGIT_SCALE,
                     'Standard deviation for logit norm')
  flags.DEFINE_float('clean_label_smoothing', CLEAN_LABEL_SMOOTHING,
                     'Amount of label smoothing on clean examples')
  flags.DEFINE_float('noisy_label_smoothing', 0.4,
                     'Amount of label smoothing on noisy examples')
  flags.DEFINE_bool('backprop_through_attack', False,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))
  flags.DEFINE_bool(
      'gcn', GCN, 'If True, use global contrast normalization on the input')
  flags.DEFINE_integer('use_input_dropout', USE_INPUT_DROPOUT,
                       'If True, apply dropout to input during training')
  flags.DEFINE_float('hid_dropout_keep_prob',
                     HID_DROPOUT_KEEP_PROB, 'Keep prob for hidden dropout')
  flags.DEFINE_string('model_name', MODEL_NAME,
                      "Name of model class from cifar10_model.py to use")

  tf.app.run()
