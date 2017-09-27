import six
import math
import time
import os
import random
import logging

import tensorflow as tf

from cleverhans.utils_tf import model_loss
from cleverhans.utils_tf import batch_indices

from model import clone_variable

from evaluator import create_adv_by_name


def build_train_op(manager, predictions, y, predictions_adv):
    model = manager.model
    hparams = manager.hparams
    if hparams.model_type == 'resnet_tf':
        build_train_op = model.build_cost
    else:
        build_train_op = model_loss

    # Define loss
    with tf.variable_scope('train_loss'):
        if predictions_adv is not None:
            if hparams.only_adv_train:
                loss = build_train_op(y, predictions_adv)
            else:
                loss = build_train_op(y, predictions)
                adv_loss = build_train_op(y, predictions_adv)
                loss = (loss + adv_loss) / 2
        else:
            loss = build_train_op(y, predictions)

    loss = tf.Print(loss, [loss])

    if hparams.model_type == 'resnet_tf':
        train_step = model.build_train_op_from_cost(loss)
    else:
        optim = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
        train_step = optim.minimize(loss)

    return [train_step]


def update_learning_params(trainer, manager):
    model = manager.model
    hparams = manager.hparams
    fd = trainer.feed_dict
    trainer.step_num = trainer.step_num

    if hparams.model_type == 'resnet_tf':
        if trainer.step_num < hparams.lrn_step:
            lrn_rate = hparams.resnet_lrn
        elif trainer.step_num < 30000:
            lrn_rate = hparams.resnet_lrn/10
        elif trainer.step_num < 35000:
            lrn_rate = hparams.resnet_lrn/100
        else:
            lrn_rate = hparams.resnet_lrn/1000

        fd[model.lrn_rate] = lrn_rate


def create_train_graph_multigpu(manager):
    assert '_multigpu' in manager.hparams.attack_type_train

    hparams = manager.hparams
    model = manager.model
    x_pre, x, y = manager.g0_inputs
    sess = manager.sess

    # Generates steps on gpus 0-(ngpu-1)
    logging.info("Initializing train attack %s" % hparams.attack_type_train)
    inputs, outputs = create_adv_by_name(
        model, x, hparams.attack_type_train,
        sess, y=y, nb_iter=hparams.attack_nb_iter_train,
        dataset=hparams.dataset, ngpu=hparams.ngpu)

    assert len(inputs) == len(outputs)
    # 0
    # inputs[0] = (x_pre, y)

    # copy y forward
    for i in range(len(outputs)):
        if i > 0:
            with tf.device(inputs[i][-1].device):
                y2 = clone_variable('y%d' % i, y)
        else:
            y2 = y
        inputs[i] = inputs[i] + (y2,)
        outputs[i] = outputs[i] + (y2,)

    # train step on last gpu
    x, adv_x, y = outputs[-1]
    device_name = '/gpu:%d' % (hparams.ngpu-1)
    model.set_device(device_name)
    with tf.device(device_name):
        with tf.variable_scope('last'):
            x2 = clone_variable('x_-1', x)
            adv2_x = clone_variable('adv_x_-1', adv_x)
            y2 = clone_variable('y_-1', y)
            inputs += [(x2, adv2_x, y2)]
            if not hparams.adv_train:
                preds = model.fprop(x2, training=True, bn_training=True)
                preds_2_adv = None
            elif not hparams.only_adv_train:
                preds = model.fprop(x2, training=True)
                preds_2_adv = model.fprop(adv2_x, training=True,
                                          bn_training=True)
            else:
                preds = None
                preds_2_adv = model.fprop(adv2_x, training=True,
                                          bn_training=True)
            train_fetches = build_train_op(manager, preds, y2, preds_2_adv)

    outputs += [train_fetches]

    device_name = '/gpu:%d' % (hparams.ngpu-1)
    model.set_device(device_name)
    with tf.device(device_name):
        sync_ops = model.create_sync_ops(host_device=device_name)

    return inputs, outputs, sync_ops


def model_train(trainer, manager):
    """
    Train a TF graph
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param hparams.save: boolean controlling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param evaluate: function that is run after each training iteration
                    (typically to display the test/validation accuracy).
    """

    hparams = manager.hparams
    batch_size = hparams.batch_size
    nb_epochs = hparams.nb_epochs
    train_dir = hparams.save_dir
    filename = 'model.ckpt'
    X_train = manager.X_train
    Y_train = manager.Y_train

    sess = manager.sess

    with sess.as_default():
        X_batch = X_train[:batch_size]
        Y_batch = Y_train[:batch_size]
        trainer.init_tf(X_batch, Y_batch)

        for epoch in six.moves.xrange(nb_epochs):
            logging.info("Epoch " + str(epoch))

            # Compute number of batches
            nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
            assert nb_batches * batch_size >= len(X_train)

            # Indices to shuffle training set
            index_shuf = list(range(len(X_train)))
            random.shuffle(index_shuf)

            prev = time.time()
            for batch in range(nb_batches):
                # Compute batch start and end indices
                start, end = batch_indices(
                    batch, len(X_train), batch_size)

                # Perform one training step
                update_learning_params(trainer, manager)

                # train step
                X_batch = X_train[index_shuf[start:end]]
                Y_batch = Y_train[index_shuf[start:end]]

                trainer.run(X_batch, Y_batch)
                trainer.sync_params()

            # clean up the queue
            while not trainer.is_finished():
                trainer.run()

            trainer.sync_params(forced=True)

            assert end >= len(X_train)  # Check that all examples were used
            cur = time.time()
            logging.info("\tEpoch took " + str(cur - prev) + " seconds")
            prev = cur

            manager.eval()

            cond = ((epoch+1) % hparams.save_steps == 0
                    or epoch == nb_epochs)
            if hparams.save and cond:
                save_path = os.path.join(train_dir, filename)
                saver = tf.train.Saver()
                saver.save(sess, save_path)
                logging.info("Completed model training and saved at:" +
                             str(save_path))
            else:
                logging.info("Completed model training.")
