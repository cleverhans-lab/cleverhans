"""
This code reproduces the MNIST results from the paper
Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning
https://arxiv.org/abs/1803.04765
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
from bisect import bisect_left
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange
import tensorflow as tf
import falconn
from cleverhans.attacks import FastGradientMethod
from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.model import Model
from cleverhans.picklable_model import MLP, Conv2D, ReLU, Flatten, Linear, Softmax
from cleverhans.train import train
from cleverhans.utils_tf import batch_eval, model_eval

if 'DISPLAY' not in os.environ:
  matplotlib.use('Agg')


FLAGS = tf.flags.FLAGS


def make_basic_picklable_cnn(nb_filters=64, nb_classes=10,
                             input_shape=(None, 28, 28, 1)):
  """The model for the picklable models tutorial.
  """
  layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
            ReLU(),
            Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
            ReLU(),
            Flatten(),
            Linear(nb_classes),
            Softmax()]
  model = MLP(layers, input_shape)
  return model


class DkNNModel(Model):
  def __init__(self, neighbors, layers, get_activations, train_data, train_labels,
               nb_classes, scope=None, nb_tables=200, number_bits=17):
    """
    Implements the DkNN algorithm. See https://arxiv.org/abs/1803.04765 for more details.

    :param neighbors: number of neighbors to find per layer.
    :param layers: a list of layer names to include in the DkNN.
    :param get_activations: a callable that takes a np array and a layer name and returns its activations on the data.
    :param train_data: a np array of training data.
    :param train_labels: a np vector of training labels.
    :param nb_classes: the number of classes in the task.
    :param scope: a TF scope that was used to create the underlying model.
    :param nb_tables: number of tables used by FALCONN to perform locality-sensitive hashing.
    :param number_bits: number of hash bits used by FALCONN.
    """
    super(DkNNModel, self).__init__(nb_classes=nb_classes, scope=scope)
    self.neighbors = neighbors
    self.nb_tables = nb_tables
    self.layers = layers
    self.get_activations = get_activations
    self.nb_cali = -1
    self.calibrated = False
    self.number_bits = number_bits

    # Compute training data activations
    self.nb_train = train_labels.shape[0]
    assert self.nb_train == train_data.shape[0]
    self.train_activations = get_activations(train_data)
    self.train_labels = train_labels

    # Build locality-sensitive hashing tables for training representations
    self.train_activations_lsh = copy.copy(self.train_activations)
    self.init_lsh()

  def init_lsh(self):
    """
    Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data.
    """
    self.query_objects = {
    }  # contains the object that can be queried to find nearest neighbors at each layer.
    # mean of training data representation per layer (that needs to be substracted before LSH).
    self.centers = {}
    for layer in self.layers:
      assert self.nb_tables >= self.neighbors

      # Normalize all the lenghts, since we care about the cosine similarity.
      self.train_activations_lsh[layer] /= np.linalg.norm(
          self.train_activations_lsh[layer], axis=1).reshape(-1, 1)

      # Center the dataset and the queries: this improves the performance of LSH quite a bit.
      center = np.mean(self.train_activations_lsh[layer], axis=0)
      self.train_activations_lsh[layer] -= center
      self.centers[layer] = center

      # LSH parameters
      params_cp = falconn.LSHConstructionParameters()
      params_cp.dimension = len(self.train_activations_lsh[layer][1])
      params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
      params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
      params_cp.l = self.nb_tables
      params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
      params_cp.seed = 5721840
      # we want to use all the available threads to set up
      params_cp.num_setup_threads = 0
      params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

      # we build 18-bit hashes so that each table has
      # 2^18 bins; this is a good choice since 2^18 is of the same
      # order of magnitude as the number of data points
      falconn.compute_number_of_hash_functions(self.number_bits, params_cp)

      print('Constructing the LSH table')
      table = falconn.LSHIndex(params_cp)
      table.setup(self.train_activations_lsh[layer])

      # Parse test feature vectors and find k nearest neighbors
      query_object = table.construct_query_object()
      query_object.set_num_probes(self.nb_tables)
      self.query_objects[layer] = query_object

  def find_train_knns(self, data_activations):
    """
    Given a data_activation dictionary that contains a np array with activations for each layer,
    find the knns in the training data.
    """
    knns_ind = {}
    knns_labels = {}

    for layer in self.layers:
      # Pre-process representations of data to normalize and remove training data mean.
      data_activations_layer = copy.copy(data_activations[layer])
      nb_data = data_activations_layer.shape[0]
      data_activations_layer /= np.linalg.norm(
          data_activations_layer, axis=1).reshape(-1, 1)
      data_activations_layer -= self.centers[layer]

      # Use FALCONN to find indices of nearest neighbors in training data.
      knns_ind[layer] = np.zeros(
          (data_activations_layer.shape[0], self.neighbors), dtype=np.int32)
      knn_errors = 0
      for i in range(data_activations_layer.shape[0]):
        query_res = self.query_objects[layer].find_k_nearest_neighbors(
            data_activations_layer[i], self.neighbors)
        try:
          knns_ind[layer][i, :] = query_res
        except:  # pylint: disable-msg=W0702
          knns_ind[layer][i, :len(query_res)] = query_res
          knn_errors += knns_ind[layer].shape[1] - len(query_res)

      # Find labels of neighbors found in the training data.
      knns_labels[layer] = np.zeros((nb_data, self.neighbors), dtype=np.int32)
      for data_id in range(nb_data):
        knns_labels[layer][data_id, :] = self.train_labels[knns_ind[layer][data_id]]

    return knns_ind, knns_labels

  def nonconformity(self, knns_labels):
    """
    Given an dictionary of nb_data x nb_classes dimension, compute the nonconformity of
    each candidate label for each data point: i.e. the number of knns whose label is
    different from the candidate label.
    """
    nb_data = knns_labels[self.layers[0]].shape[0]
    knns_not_in_class = np.zeros((nb_data, self.nb_classes), dtype=np.int32)
    for i in range(nb_data):
      # Compute number of nearest neighbors per class
      knns_in_class = np.zeros(
          (len(self.layers), self.nb_classes), dtype=np.int32)
      for layer_id, layer in enumerate(self.layers):
        knns_in_class[layer_id, :] = np.bincount(
            knns_labels[layer][i], minlength=self.nb_classes)

      # Compute number of knns in other class than class_id
      for class_id in range(self.nb_classes):
        knns_not_in_class[i, class_id] = np.sum(
            knns_in_class) - np.sum(knns_in_class[:, class_id])
    return knns_not_in_class

  def preds_conf_cred(self, knns_not_in_class):
    """
    Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
    the DkNN's prediction, confidence and credibility.
    """
    nb_data = knns_not_in_class.shape[0]
    preds_knn = np.zeros(nb_data, dtype=np.int32)
    confs = np.zeros((nb_data, self.nb_classes), dtype=np.float32)
    creds = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

    for i in range(nb_data):
      # p-value of test input for each class
      p_value = np.zeros(self.nb_classes, dtype=np.float32)

      for class_id in range(self.nb_classes):
        # p-value of (test point, candidate label)
        p_value[class_id] = (float(self.nb_cali) - bisect_left(
            self.cali_nonconformity, knns_not_in_class[i, class_id])) / float(self.nb_cali)

      preds_knn[i] = np.argmax(p_value)
      confs[i, preds_knn[i]] = 1. - p_value[np.argsort(p_value)[-2]]
      creds[i, preds_knn[i]] = p_value[preds_knn[i]]
    return preds_knn, confs, creds

  def fprop_np(self, data_np):
    """
    Performs a forward pass through the DkNN on an numpy array of data.
    """
    if not self.calibrated:
      raise ValueError(
          "DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring.")
    data_activations = self.get_activations(data_np)
    _, knns_labels = self.find_train_knns(data_activations)
    knns_not_in_class = self.nonconformity(knns_labels)
    _, _, creds = self.preds_conf_cred(knns_not_in_class)
    return creds

  def fprop(self, x):
    """
    Performs a forward pass through the DkNN on a TF tensor by wrapping
    the fprop_np method.
    """
    logits = tf.py_func(self.fprop_np, [x], tf.float32)
    return {self.O_LOGITS: logits}

  def calibrate(self, cali_data, cali_labels):
    """
    Runs the DkNN on holdout data to calibrate the credibility metric.
    :param cali_data: np array of calibration data.
    :param cali_labels: np vector of calibration labels.
    """
    self.nb_cali = cali_labels.shape[0]
    self.cali_activations = self.get_activations(cali_data)
    self.cali_labels = cali_labels

    print("Starting calibration of DkNN.")
    cali_knns_ind, cali_knns_labels = self.find_train_knns(
        self.cali_activations)
    assert all([v.shape == (self.nb_cali, self.neighbors)
                for v in cali_knns_ind.itervalues()])
    assert all([v.shape == (self.nb_cali, self.neighbors)
                for v in cali_knns_labels.itervalues()])

    cali_knns_not_in_class = self.nonconformity(cali_knns_labels)
    cali_knns_not_in_l = np.zeros(self.nb_cali, dtype=np.int32)
    for i in range(self.nb_cali):
      cali_knns_not_in_l[i] = cali_knns_not_in_class[i, cali_labels[i]]
    cali_knns_not_in_l_sorted = np.sort(cali_knns_not_in_l)
    self.cali_nonconformity = np.trim_zeros(cali_knns_not_in_l_sorted, trim='f')
    self.nb_cali = self.cali_nonconformity.shape[0]
    self.calibrated = True
    print("DkNN calibration complete.")


def plot_reliability_diagram(confidence, labels, filepath):
  """
  Takes in confidence values for predictions and correct
  labels for the data, plots a reliability diagram.
  :param confidence: nb_samples x nb_classes (e.g., output of softmax)
  :param labels: vector of nb_samples
  :param filepath: where to save the diagram
  :return:
  """
  assert len(confidence.shape) == 2
  assert len(labels.shape) == 1
  assert confidence.shape[0] == labels.shape[0]
  print('Saving reliability diagram at: ' + str(filepath))
  if confidence.max() <= 1.:
        # confidence array is output of softmax
    bins_start = [b / 10. for b in xrange(0, 10)]
    bins_end = [b / 10. for b in xrange(1, 11)]
    bins_center = [(b + .5) / 10. for b in xrange(0, 10)]
    preds_conf = np.max(confidence, axis=1)
    preds_l = np.argmax(confidence, axis=1)
  else:
    raise ValueError('Confidence values go above 1.')

  print(preds_conf.shape, preds_l.shape)

  # Create var for reliability diagram
  # Will contain mean accuracies for each bin
  reliability_diag = []
  num_points = []  # keeps the number of points in each bar

  # Find average accuracy per confidence bin
  for bin_start, bin_end in zip(bins_start, bins_end):
    above = preds_conf >= bin_start
    if bin_end == 1.:
      below = preds_conf <= bin_end
    else:
      below = preds_conf < bin_end
    mask = np.multiply(above, below)
    num_points.append(np.sum(mask))
    bin_mean_acc = max(0, np.mean(preds_l[mask] == labels[mask]))
    reliability_diag.append(bin_mean_acc)

  # Plot diagram
  assert len(reliability_diag) == len(bins_center)
  print(reliability_diag)
  print(bins_center)
  print(num_points)
  fig, ax1 = plt.subplots()
  _ = ax1.bar(bins_center, reliability_diag, width=.1, alpha=0.8)
  plt.xlim([0, 1.])
  ax1.set_ylim([0, 1.])

  ax2 = ax1.twinx()
  print(sum(num_points))
  ax2.plot(bins_center, num_points, color='r', linestyle='-', linewidth=7.0)
  ax2.set_ylabel('Number of points in the data', fontsize=16, color='r')

  if len(np.argwhere(confidence[0] != 0.)) == 1:
    # This is a DkNN diagram
    ax1.set_xlabel('Prediction Credibility', fontsize=16)
  else:
    # This is a softmax diagram
    ax1.set_xlabel('Prediction Confidence', fontsize=16)
  ax1.set_ylabel('Prediction Accuracy', fontsize=16)
  ax1.tick_params(axis='both', labelsize=14)
  ax2.tick_params(axis='both', labelsize=14, colors='r')
  fig.tight_layout()
  plt.savefig(filepath, bbox_inches='tight')


def dknn_tutorial():
  # Get MNIST data.
  mnist = MNIST()
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # Use Image Parameters.
  img_rows, img_cols, nchannels = x_train.shape[1:4]
  nb_classes = y_train.shape[1]

  with tf.Session() as sess:
    with tf.variable_scope('dknn'):
      # Define input TF placeholder.
      x = tf.placeholder(tf.float32, shape=(
          None, img_rows, img_cols, nchannels))
      y = tf.placeholder(tf.float32, shape=(None, nb_classes))

      # Define a model.
      model = make_basic_picklable_cnn()
      preds = model.get_logits(x)
      loss = CrossEntropy(model, smoothing=0.)

      # Define the test set accuracy evaluation.
      def evaluate():
        acc = model_eval(sess, x, y, preds, x_test, y_test,
                         args={'batch_size': FLAGS.batch_size})
        print('Test accuracy on test examples: %0.4f' % acc)

      # Train the model
      train_params = {'nb_epochs': FLAGS.nb_epochs,
                      'batch_size': FLAGS.batch_size, 'learning_rate': FLAGS.lr}
      train(sess, loss, x_train, y_train, evaluate=evaluate,
            args=train_params, var_list=model.get_params())

      # Define callable that returns a dictionary of all activations for a dataset
      def get_activations(data):
        data_activations = {}
        for layer in layers:
          layer_sym = tf.layers.flatten(model.get_layer(x, layer))
          data_activations[layer] = batch_eval(sess, [x], [layer_sym], [data],
                                               args={'batch_size': FLAGS.batch_size})[0]
        return data_activations

      # Use a holdout of the test set to simulate calibration data for the DkNN.
      train_data = x_train
      train_labels = np.argmax(y_train, axis=1)
      cali_data = x_test[:FLAGS.nb_cali]
      y_cali = y_test[:FLAGS.nb_cali]
      cali_labels = np.argmax(y_cali, axis=1)
      test_data = x_test[FLAGS.nb_cali:]
      y_test = y_test[FLAGS.nb_cali:]

      # Extract representations for the training and calibration data at each layer of interest to the DkNN.
      layers = ['ReLU1', 'ReLU3', 'ReLU5', 'logits']

      # Wrap the model into a DkNNModel
      dknn = DkNNModel(FLAGS.neighbors, layers, get_activations,
                       train_data, train_labels, nb_classes, scope='dknn')
      dknn.calibrate(cali_data, cali_labels)

      # Generate adversarial examples
      fgsm = FastGradientMethod(model, sess=sess)
      attack_params = {'eps': .25, 'clip_min': 0., 'clip_max': 1.}
      adv = sess.run(fgsm.generate(x, **attack_params),
                     feed_dict={x: test_data})

      # Test the DkNN on clean test data and FGSM test data
      for data_in, fname in zip([test_data, adv], ['test', 'adv']):
        dknn_preds = dknn.fprop_np(data_in)
        print(dknn_preds.shape)
        print(np.mean(np.argmax(dknn_preds, axis=1) == np.argmax(y_test, axis=1)))
        plot_reliability_diagram(dknn_preds, np.argmax(
            y_test, axis=1), '/tmp/dknn_' + fname + '.pdf')

  return True


def main(argv=None):
  assert dknn_tutorial()


if __name__ == '__main__':
  tf.flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
  tf.flags.DEFINE_integer('batch_size', 500, 'Size of training batches')
  tf.flags.DEFINE_float('lr', 0.001, 'Learning rate for training')

  tf.flags.DEFINE_integer(
      'nb_cali', 750, 'Number of calibration points for the DkNN')
  tf.flags.DEFINE_integer(
      'neighbors', 75, 'Number of neighbors per layer for the DkNN')

  tf.app.run()
