"""
This model shows how to train a model with Soft Nearest Neighbor Loss
regularization. The paper which presents this method can be found at
https://arxiv.org/abs/1902.01889
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

from cleverhans.compat import flags
from cleverhans.loss import SNNLCrossEntropy, CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model_zoo.soft_nearest_neighbor_loss.SNNL_regularized_model import ModelBasicCNN

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NB_FILTERS = 64
SNNL_FACTOR = -10.0
OUTPUT_DIR = '/tmp/'


def SNNL_example(train_start=0, train_end=60000, test_start=0,
                 test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE,
                 nb_filters=NB_FILTERS,
                 SNNL_factor=SNNL_FACTOR,
                 output_dir=OUTPUT_DIR):
  """
  A simple model trained to minimize Cross Entropy and Maximize Soft Nearest
  Neighbor Loss at each internal layer. This outputs a TSNE of the sign of
  the adversarial gradients of a trained model. A model with a negative
  SNNL_factor will show little or no class clusters, while a model with a
  0 SNNL_factor will have class clusters in the adversarial gradient direction.
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param SNNL_factor: multiplier for Soft Nearest Neighbor Loss
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  sess = tf.Session()

  # Get MNIST data
  mnist = MNIST(train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
  }
  eval_params = {'batch_size': batch_size}
  rng = np.random.RandomState([2017, 8, 30])

  def do_eval(preds, x_set, y_set, report_key):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    print('Test accuracy on legitimate examples: %0.4f' % (acc))

  model = ModelBasicCNN('model', nb_classes, nb_filters)
  preds = model.get_logits(x)
  cross_entropy_loss = CrossEntropy(model)
  if not SNNL_factor:
    loss = cross_entropy_loss
  else:
    loss = SNNLCrossEntropy(model, factor=SNNL_factor,
                            optimize_temperature=False)

  def evaluate():
    do_eval(preds, x_test, y_test, 'clean_train_clean_eval')

  train(sess, loss, x_train, y_train, evaluate=evaluate,
        args=train_params, rng=rng, var_list=model.get_params())

  do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

  def imscatter(points, images, ax=None, zoom=1, cmap="hot"):
    if ax is None:
      ax = plt.gca()
    artists = []
    i = 0
    if not isinstance(cmap, list):
      cmap = [cmap] * len(points)
    for x0, y0 in points:
      transformed = (images[i] - np.min(images[i])) / \
          (np.max(images[i]) - np.min(images[i]))
      im = OffsetImage(transformed[:, :, 0], zoom=zoom, cmap=cmap[i])
      ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
      artists.append(ax.add_artist(ab))
      i += 1
    ax.update_datalim(np.column_stack(np.transpose(points)))
    ax.autoscale()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return artists

  adv_grads = tf.sign(tf.gradients(cross_entropy_loss.fprop(x, y), x))
  feed_dict = {x: x_test[:batch_size], y: y_test[:batch_size]}
  adv_grads_val = sess.run(adv_grads, feed_dict=feed_dict)
  adv_grads_val = np.reshape(adv_grads_val, (batch_size, img_rows * img_cols))

  X_embedded = TSNE(n_components=2, verbose=0).fit_transform(adv_grads_val)
  plt.figure(num=None, figsize=(50, 50), dpi=40, facecolor='w', edgecolor='k')
  plt.title("TSNE of Sign of Adv Gradients, SNNLCrossEntropy Model, factor:" +
            str(FLAGS.SNNL_factor), fontsize=42)
  imscatter(X_embedded, x_test[:batch_size], zoom=2, cmap="Purples")
  plt.savefig(output_dir + 'adversarial_gradients_SNNL_factor_' +
              str(SNNL_factor) + '.png')


def main(argv=None):
  SNNL_example(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
               learning_rate=FLAGS.learning_rate,
               nb_filters=FLAGS.nb_filters,
               SNNL_factor=FLAGS.SNNL_factor,
               output_dir=FLAGS.output_dir)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_filters', NB_FILTERS,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('SNNL_factor', SNNL_FACTOR,
                     'Multiplier for Soft Nearest Neighbor Loss')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')
  flags.DEFINE_string('output_dir', OUTPUT_DIR,
                      'output directory for saving figures')

  tf.app.run()
