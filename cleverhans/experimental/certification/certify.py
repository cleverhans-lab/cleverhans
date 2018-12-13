"""Code for running the certification problem."""

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
flags.DEFINE_string('init_dual_file', None,
                    'Path of numpy file with dual variables to initialize')
flags.DEFINE_string('test_input', None,
                    'Path of numpy file with test input to certify')
flags.DEFINE_integer('true_class', 0,
                     'True class of the test input')
flags.DEFINE_integer('adv_class', -1,
                     'target class of adversarial example; all classes if -1')
flags.DEFINE_float('input_minval', -1,
                   'Minimum value of valid input')
flags.DEFINE_float('input_maxval', 1,
                   'Maximum value of valid input')
flags.DEFINE_float('epsilon', 0.2,
                   'Size of perturbation')
# Nu might need tuning based on the network
flags.DEFINE_float('init_nu', 300.0,
                   'Initialization of nu variable.')
flags.DEFINE_float('init_penalty', 100.0,
                   'Initial penalty')
flags.DEFINE_integer('small_eig_num_steps', 500,
                     'Number of eigen value steps in intermediate iterations')
flags.DEFINE_integer('large_eig_num_steps', 5000,
                     'Number of eigen value steps in each outer iteration')
flags.DEFINE_integer('inner_num_steps', 600,
                     'Number of steps to run in inner loop')
flags.DEFINE_float('outer_num_steps', 10,
                   'Number of steps to run in outer loop')
flags.DEFINE_float('beta', 2,
                   'Multiplicative factor to increase penalty by')
flags.DEFINE_float('smoothness_parameter', 0.001,
                   'Smoothness parameter if using eigen decomposition')
flags.DEFINE_float('eig_learning_rate', 0.001,
                   'Learning rate for computing min eigen value')
flags.DEFINE_string('optimizer', 'adam',
                    'Optimizer to use for entire optimization')
flags.DEFINE_float('init_learning_rate', 0.1,
                   'Initial learning rate')
flags.DEFINE_float('learning_rate_decay', 0.1,
                   'Decay of learning rate')
flags.DEFINE_float('momentum_parameter', 0.9,
                   'Momentum parameter if using momentum optimizer')
flags.DEFINE_integer('print_stats_steps', 50,
                     'Number of steps to print stats after')
flags.DEFINE_string('stats_folder', None,
                    'Folder to save stats of the iterations')
flags.DEFINE_integer('projection_steps', 200,
                     'Number of steps to compute projection after')
flags.DEFINE_integer('num_classes', 10,
                     'Total number of classes')
flags.DEFINE_enum('verbosity', 'INFO',
                  ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                  'Logging verbosity level.')


def main(_):
  tf.logging.set_verbosity(FLAGS.verbosity)

  net_weights, net_biases, net_layer_types = read_weights.read_weights(
      FLAGS.checkpoint, FLAGS.model_json)
  nn_params = neural_net_params.NeuralNetParams(
      net_weights, net_biases, net_layer_types)
  tf.logging.info('Loaded neural network with size of layers: %s',
                  nn_params.sizes)
  dual_var = utils.initialize_dual(nn_params, FLAGS.init_dual_file,
                                   init_nu=FLAGS.init_nu)
  # Reading test input and reshaping
  with tf.gfile.Open(FLAGS.test_input) as f:
    test_input = np.load(f)
  test_input = np.reshape(test_input, [np.size(test_input), 1])

  if FLAGS.adv_class == -1:
    start_class = 0
    end_class = FLAGS.num_classes
  else:
    start_class = FLAGS.adv_class
    end_class = FLAGS.adv_class + 1
  for adv_class in range(start_class, end_class):
    tf.logging.info('Running certification for adversarial class %d', adv_class)
    if adv_class == FLAGS.true_class:
      continue
    dual = dual_formulation.DualFormulation(dual_var,
                                            nn_params,
                                            test_input,
                                            FLAGS.true_class,
                                            adv_class,
                                            FLAGS.input_minval,
                                            FLAGS.input_maxval,
                                            FLAGS.epsilon)
    dual.set_differentiable_objective()
    dual.get_full_psd_matrix()
    optimization_params = {
        'init_penalty': FLAGS.init_penalty,
        'large_eig_num_steps': FLAGS.large_eig_num_steps,
        'small_eig_num_steps': FLAGS.small_eig_num_steps,
        'inner_num_steps': FLAGS.inner_num_steps,
        'outer_num_steps': FLAGS.outer_num_steps,
        'beta': FLAGS.beta,
        'smoothness_parameter': FLAGS.smoothness_parameter,
        'eig_learning_rate': FLAGS.eig_learning_rate,
        'optimizer': FLAGS.optimizer,
        'init_learning_rate': FLAGS.init_learning_rate,
        'learning_rate_decay': FLAGS.learning_rate_decay,
        'momentum_parameter': FLAGS.momentum_parameter,
        'print_stats_steps': FLAGS.print_stats_steps,
        'stats_folder': FLAGS.stats_folder,
        'projection_steps': FLAGS.projection_steps}
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      optimization_object = optimization.Optimization(dual,
                                                      sess,
                                                      optimization_params)
      optimization_object.prepare_one_step()
      is_cert_found = optimization_object.run_optimization()
      if not is_cert_found:
        print('Example could not be verified')
        exit()
  print('Example successfully verified')

if __name__ == '__main__':
  tf.app.run(main)
