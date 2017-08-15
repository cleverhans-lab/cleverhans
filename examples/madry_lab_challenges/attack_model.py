"""Runs CleverHans attacks on the Madry Lab MNIST challenge model

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from madry_mnist_model import MadryMNIST
from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_mnist import data_mnist


FLAGS = flags.FLAGS


def main(argv):
    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    if checkpoint is None:
        raise ValueError("Couldn't find latest checkpoint in " +
                         FLAGS.checkpoint_dir)

    train_start = 0
    train_end = 60000
    test_start = 0
    test_end = 10000
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    assert Y_train.shape[1] == 10

    # NOTE: for compatibility with Madry Lab downloadable checkpoints,
    # we cannot enclose this in a scope or do anything else that would
    # change the automatic naming of the variables.
    model = MadryMNIST()

    x_input = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    if FLAGS.attack_type == 'fgsm':
        fgsm = FastGradientMethod(model)
        fgsm_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.}
        adv_x = fgsm.generate(x_image, **fgsm_params)
    elif FLAGS.attack_type == 'bim':
        bim = BasicIterativeMethod(model)
        bim_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.,
                      'nb_iter': 50,
                      'eps_iter': .01}
        adv_x = bim.generate(x_image, **bim_params)
    else:
        raise ValueError(FLAGS.attack_type)
    preds_adv = model.get_probs(adv_x)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, checkpoint)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': FLAGS.batch_size}
        t1 = time.time()
        acc = model_eval(
            sess, x_image, y, preds_adv, X_test, Y_test, args=eval_par)
        t2 = time.time()
        print("Took", t2 - t1, "seconds")
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)


if __name__ == '__main__':

    dirs = ['models', 'adv_trained']
    if "MNIST_CHALLENGE_DIR" in os.environ:
        dirs.insert(0, os.environ['MNIST_CHALLENGE_DIR'])
    default_checkpoint_dir = os.path.join(*dirs)

    flags.DEFINE_integer('batch_size', 128, "batch size")
    flags.DEFINE_float(
        'label_smooth', 0.1, ("Amount to subtract from correct label "
                              "and distribute among other labels"))
    flags.DEFINE_string(
        'attack_type', 'fgsm', ("Attack type: 'fgsm'->fast gradient sign"
                                "method, 'bim'->'basic iterative method'"))
    flags.DEFINE_string('checkpoint_dir', default_checkpoint_dir,
                        'Checkpoint directory to load')
    app.run(main)
