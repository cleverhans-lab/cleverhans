"""
Copy-paste of mnist_l1rbf.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from tensorflow.python.platform import app
from tensorflow.python.platform import flags


from manager import Manager
from trainer import model_train, create_train_graph_multigpu
from runner import RunnerMultiGPU


FLAGS = flags.FLAGS


def mnist_tutorial(hparams=None):
    """
    """

    manager = Manager(hparams)

    inputs, outputs, sync_ops = create_train_graph_multigpu(manager)
    trainer = RunnerMultiGPU(inputs, outputs, sync_ops, manager)
    model_train(trainer, manager)
    manager.evaluate(inc_epoch=False)

    return manager.finish()


def main(argv=None):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    hparams = flags.FLAGS

    mnist_tutorial(hparams=hparams)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_boolean('adv_train', False,
                         'Whether to do adversarial training')
    flags.DEFINE_boolean('save', True,
                         'Whether to save from a checkpoint')
    flags.DEFINE_string('save_dir', 'runs/X',
                        'Location to store logs/model.')
    flags.DEFINE_string('model_type', 'madry',
                        'Model type: madry|resnet_tf')
    flags.DEFINE_string('attack_type_train', 'PGDpgd_y_multigpu',
                        'Attack type for adversarial training:\
                        FGSM|PGDpgd_y')
    flags.DEFINE_string('attack_type_test', 'PGDpgd_y',
                        'Attack type for test: FGSM|PGDpgd_y')
    flags.DEFINE_boolean('no_extra_tests', False,
                         'Disable some tests for debugging.')
    flags.DEFINE_string('dataset', 'mnist', 'Dataset mnist|cifar10')
    flags.DEFINE_boolean('only_adv_train', False,
                         'Do not train with clean examples when adv training.')
    flags.DEFINE_integer('save_steps', 50, 'Save model per X steps.')
    flags.DEFINE_integer('attack_nb_iter_train', None,
                         'Number of iterations of training attack')
    flags.DEFINE_integer('eval_iters', 1, 'Evaluate every X steps')
    flags.DEFINE_integer('lrn_step', 30000, 'Step to decrease learning rate')
    flags.DEFINE_float('resnet_lrn', .1, 'Initial learning rate for resnet')
    flags.DEFINE_string('optimizer', 'mom', 'Optimizer for resnet')
    flags.DEFINE_integer('ngpu', 1, 'Number of gpus')
    flags.DEFINE_integer('sync_step', 1, 'Sync params frequency')
    flags.DEFINE_boolean('fast_tests', False, 'Fast tests against attacks')

    app.run()
