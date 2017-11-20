import logging
from collections import OrderedDict

import tensorflow as tf

from model import clone_variable

from evaluator import create_adv_by_name
from trainer import TrainManager
from runner import RunnerMultiGPU


class TrainerMultiGPU(TrainManager):
    def __init__(self, *args, **kwargs):
        super(TrainerMultiGPU, self).__init__(*args, **kwargs)
        self.runner = RunnerMultiGPU(self.inputs, self.outputs, sess=self.sess)

    def clone_g0_inputs_on_ngpus(self, inputs, outputs, g0_inputs):
        """
        Clone variables unused by the attack on all GPUs. Specifically, the
        ground-truth label, y, has to be preserved until the training step.

        :param inputs: A list of dictionaries as the inputs to each step.
        :param outputs: A list of dictionaries as the outputs of each step.
        :param g0_inputs: Initial variables to be cloned.
        :return: Updated inputs and outputs.
        """
        assert len(inputs) == len(outputs), (
            'Inputs and outputs should have the same number of elements.')

        inputs[0].update(g0_inputs)
        outputs[0].update(g0_inputs)

        # Copy g0_inputs forward
        for i in range(1, len(inputs)):
            # Create the graph for i'th step of attack
            device_name = inputs[i]['x'].device
            with tf.device(device_name):
                with tf.variable_scope('step%d' % i):
                    for k, v in g0_inputs.iteritems():
                        if k not in inputs[i]:
                            v_copy = clone_variable(k, v)
                            inputs[i][k] = v_copy
                            outputs[i][k] = v_copy

        return inputs, outputs

    def create_train_graph(self):
        super(TrainerMultiGPU, self).create_train_graph()
        assert '_multigpu' in self.hparams.attack_type_train

        hparams = self.hparams
        model = self.model
        sess = self.sess

        # Create trainable variables on last gpu.
        # Variables are set to trainable or non-trainable first time they are
        # created. This caused a bug when the last gpu is used both for attack
        # generation and training. With this bug the result of naive training
        # was affected by the length of the unused adversarial generation
        # graph.
        device_name = '/gpu:%d' % (hparams.ngpu-1)
        model.set_device(device_name)
        with tf.device(device_name):
            x = clone_variable('x', self.g0_inputs['x'])
            model.set_training(training=True)
            preds = model.get_probs(x)

        # Generates steps on gpus
        model.set_training(training=False)
        logging.info("Initializing train attack %s" %
                     hparams.attack_type_train)
        inputs, outputs = create_adv_by_name(
            model, self.g0_inputs['x'], hparams.attack_type_train,
            sess, y=self.g0_inputs['y'], nb_iter=hparams.attack_nb_iter_train,
            dataset=hparams.dataset, ngpu=hparams.ngpu)

        inputs, outputs = self.clone_g0_inputs_on_ngpus(
            inputs, outputs, self.g0_inputs)

        # Train step on last gpu
        device_name = '/gpu:%d' % (hparams.ngpu-1)
        model.set_device(device_name)
        with tf.device(device_name):
            with tf.variable_scope('last'):
                inputs += [OrderedDict()]
                for k, v in outputs[-1].iteritems():
                    v_copy = clone_variable(k, v)
                    inputs[-1][k] = v_copy
                x = inputs[-1]['x']
                adv_x = inputs[-1]['adv_x']
                y = inputs[-1]['y']
                if not hparams.adv_train:
                    model.set_training(training=True)
                    preds = model.get_probs(x)
                    preds_adv = None
                elif not hparams.only_adv_train:
                    model.set_training(training=True)
                    preds = model.get_probs(x)
                    model.set_training(training=True)
                    preds_adv = model.get_probs(adv_x)
                else:
                    preds = None
                    model.set_training(training=True)
                    preds_adv = model.get_probs(adv_x)
                train_fetches = self.build_train_op(preds, y, preds_adv)

        outputs += [{'fetches': train_fetches}]

        # Create the sync operation
        device_name = '/gpu:%d' % (hparams.ngpu-1)
        model.set_device(device_name)
        with tf.device(device_name):
            sync_ops = model.create_sync_ops(host_device=device_name)

        self.inputs = inputs
        self.outputs = outputs
        self.sync_ops = sync_ops

    def sync_params(self, forced=False):
        if forced or (self.step_num % self.hparams.sync_step == 0):
            self.sess.run(self.sync_ops)
