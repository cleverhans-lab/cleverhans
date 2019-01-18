"""Code for just computing the certificate (to be merged with optimization code eventually."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

from cleverhans.experimental.certification import dual_formulation
from cleverhans.experimental.certification import neural_net_params
from cleverhans.experimental.certification import optimization
from cleverhans.experimental.certification import read_weights
from cleverhans.experimental.certification import utils


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint', None,
                    'Path of checkpoint with trained model to verify')
flags.DEFINE_string('model_json', None,
                    'Path of json file with model description')
flags.DEFINE_string('model_logits', None, 
                    'Path of clean logits')
flags.DEFINE_integer('input_index', None, 
                    'Index of test input (to compute logits)')
flags.DEFINE_string('dual_folder', None,
                    'Path of numpy file with dual variables to initialize')
flags.DEFINE_string('test_input', None,
                    'Path of numpy file with test input to certify')
flags.DEFINE_integer('true_class', 0,
                     'True class of the test input')
flags.DEFINE_integer('adv_class', -1,
                     'target class of adversarial example; all classes if -1')
flags.DEFINE_float('input_minval', 0,
                   'Minimum value of valid input')
flags.DEFINE_float('input_maxval', 1,
                   'Maximum value of valid input')
flags.DEFINE_float('epsilon', 0.2,
                   'Size of perturbation')
lags.DEFINE_enum('verbosity', 'DEBUG',
                  ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                  'Logging verbosity level.')

                  dataset = 'MNIST'

def main(_):
  tf.logging.set_verbosity(FLAGS.verbosity)

  # Reading test input and reshaping
  with tf.gfile.Open(FLAGS.test_input) as f:
    test_input = np.load(f)
  test_input = np.reshape(test_input, [-1, 1])

  if(dataset=='MNIST'):
    num_rows = 28
    num_columns = 28
    num_channels = 1

  net_weights, net_biases, net_layer_types = read_weights.read_weights(
      FLAGS.checkpoint, FLAGS.model_json)
  input_shape = [num_rows, num_columns, num_channels]
  nn_params = neural_net_params.NeuralNetParams(
      net_weights, net_biases, net_layer_types, input_shape)

  # Creating feedforward object for debugging 
  net_weights, net_biases, net_layer_types = read_weights.read_weights(
    FLAGS.checkpoint, FLAGS.model_json, [num_rows, num_columns, num_channels], 
    CONV2FF=True)
  nn_params_ff = neural_net_params.NeuralNetParams(
    net_weights, net_biases, net_layer_types)


  tf.logging.info('Loaded neural network with size of layers: %s',
                  nn_params.sizes)

  tf.logging.info('Running certification for adversarial class %d', FLAGS.adv_class)
  config = tf.ConfigProto(
      device_count={'GPU':0})
  
  with tf.Session() as sess:
  	sess.run(tf.global_variables_initializer())
  	# Checking that all the weights are loaded correctly 
    nn_test_output = nn_params.nn_output(test_input, FLAGS.true_class, adv_class)
    current_test_output = sess.run(nn_test_output)
    test_logits = np.load(FLAGS.model_logits)
    true_test_output = (test_logits[FLAGS.input_index, adv_class] - 
      test_logits[FLAGS.input_index, FLAGS.true_class])
    print("True test output", true_test_output)
    if(np.abs(true_test_output - current_test_output) > 1E-3):
      print('Forward passes do not match with difference ', np.abs(true_test_output - current_test_output))
      exit()
    dual_placeholder_ff = dual_formulation.DualFormulation(self.sess,
       nn_params_ff,
       self.dual_object.test_input,
       self.dual_object.true_class,
       self.dual_object.adv_class,
       self.dual_object.input_minval,
       self.dual_object.input_maxval,
       self.dual_object.epsilon
       )
    dual_placeholder_ff.intialize_placeholders()
    dual_placeholder_ff.set_differentiable_objective()
    dual_placeholder_ff.get_full_psd_matrix()
    certificate = dual_placeholder_ff.compute_certifcate(FLAGS.dual_folder)
    print("Computed certificate", certificate)

if __name__ == 'main':
  tf.app.run(main)
