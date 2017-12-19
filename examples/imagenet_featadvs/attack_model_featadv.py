"""
Runs Feature Adversaries on a basic imagenet CNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.attacks import FastFeatureAdversaries
from model import make_imagenet_cnn


FLAGS = flags.FLAGS


def main(argv):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    input_shape = [FLAGS.batch_size, 224, 224, 3]
    x_src = tf.abs(tf.random_uniform(input_shape, 0., 1.))
    x_guide = tf.abs(tf.random_uniform(input_shape, 0., 1.))
    print("Input shape:")
    print(input_shape)

    model = make_imagenet_cnn(input_shape)
    print("Model:")
    for i, layer in enumerate(model.layers):
        print('%s %s' % (model.layer_names[i], layer.output_shape))
    attack = FastFeatureAdversaries(model)
    attack_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1.,
                     'nb_iter': FLAGS.nb_iter, 'eps_iter': 0.01,
                     'layer': FLAGS.layer}
    x_adv = attack.generate(x_src, x_guide, **attack_params)
    h_adv = model.fprop(x_adv)[FLAGS.layer]
    h_src = model.fprop(x_src)[FLAGS.layer]
    h_guide = model.fprop(x_guide)[FLAGS.layer]

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        ha, hs, hg, xa, xs, xg = sess.run(
            [h_adv, h_src, h_guide, x_adv, x_src, x_guide])

        print("L2 distance between source and adversarial example `%s`: %.4f" %
              (FLAGS.layer, ((hs-ha)*(hs-ha)).sum()))
        print("L2 distance between guide and adversarial example `%s`: %.4f" %
              (FLAGS.layer, ((hg-ha)*(hg-ha)).sum()))
        print("L2 distance between source and guide `%s`: %.4f" %
              (FLAGS.layer, ((hg-hs)*(hg-hs)).sum()))
        print("Maximum perturbation: %.4f" % np.abs((xa-xs)).max())
        print("Original features: ")
        print(hs[:10, :10])
        print("Adversarial features: ")
        print(ha[:10, :10])


if __name__ == '__main__':
    flags.DEFINE_integer('batch_size', 128, "batch size")
    flags.DEFINE_string('layer', 'logits', 'Layer for feature adversaries')
    flags.DEFINE_integer('nb_iter', 100, 'Number of iterations for the attack')
    app.run(main)
