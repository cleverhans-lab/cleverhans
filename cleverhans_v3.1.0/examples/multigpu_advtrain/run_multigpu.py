"""
This script adversarially trains a model using iterative attacks on multiple
GPUs.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from collections import namedtuple

from cleverhans.compat import app, flags

from trainer import TrainerMultiGPU
from trainer import TrainerSingleGPU


def run_trainer(hparams):
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

  if 'multigpu' in hparams.attack_type_train:
    logging.info('Multi GPU Trainer.')
    trainer = TrainerMultiGPU(hparams)
  else:
    logging.info('Single GPU Trainer.')
    trainer = TrainerSingleGPU(hparams)
  trainer.model_train()
  trainer.eval(inc_epoch=False)

  return trainer.finish()


def main(argv=None):
  f = {x: flags.FLAGS[x].value for x in dir(flags.FLAGS)}
  HParams = namedtuple('HParams', f.keys())
  hparams = HParams(**f)
  run_trainer(hparams)


if __name__ == '__main__':
  flags.DEFINE_integer('train_start', 0,
                       'Index of first training set example.')
  flags.DEFINE_integer('train_end', 60000,
                       'Index of last training set example.')
  flags.DEFINE_integer('test_start', 0,
                       'Index of first test set example.')
  flags.DEFINE_integer('test_end', 10000,
                       'Index of last test set example.')
  flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model.')
  flags.DEFINE_integer('batch_size', 128, 'Size of training batches.')
  flags.DEFINE_boolean('adv_train', False,
                       'Whether to do adversarial training.')
  flags.DEFINE_boolean('save', True,
                       'Whether to save from a checkpoint.')
  flags.DEFINE_string('save_dir', 'runs/X',
                      'Location to store logs/model.')
  flags.DEFINE_string('model_type', 'madry',
                      'Model type: basic|madry|resnet_tf.')
  flags.DEFINE_string('attack_type_train', 'MadryEtAl_y_multigpu',
                      'Attack type for adversarial training:\
                        FGSM|MadryEtAl{,_y}{,_multigpu}.')
  flags.DEFINE_string('attack_type_test', 'FGSM',
                      'Attack type for test: FGSM|MadryEtAl{,_y}.')
  flags.DEFINE_string('dataset', 'mnist', 'Dataset mnist|cifar10.')
  flags.DEFINE_boolean('only_adv_train', False,
                       'Do not train with clean examples when adv training.')
  flags.DEFINE_integer('save_steps', 50, 'Save model per X steps.')
  flags.DEFINE_integer('attack_nb_iter_train', None,
                       'Number of iterations of training attack.')
  flags.DEFINE_integer('eval_iters', 1, 'Evaluate every X steps.')
  flags.DEFINE_integer('lrn_step', 30000, 'Step to decrease learning rate'
                       'for ResNet.')
  flags.DEFINE_float('adam_lrn', 0.001, 'Learning rate for Adam Optimizer.')
  flags.DEFINE_float('mom_lrn', 0.1,
                     'Learning rate for Momentum Optimizer.')
  flags.DEFINE_integer('ngpu', 1, 'Number of gpus.')
  flags.DEFINE_integer('sync_step', 1, 'Sync params frequency.')
  flags.DEFINE_boolean('fast_tests', False, 'Fast tests against attacks.')
  flags.DEFINE_string('data_path', './datasets/', 'Path to datasets.'
                      'Each dataset should be in a subdirectory.')

  app.run()
