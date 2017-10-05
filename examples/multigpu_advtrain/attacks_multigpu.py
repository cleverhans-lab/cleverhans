import tensorflow as tf

from cleverhans.attacks import MadryEtAl

from model import clone_variable


class MadryEtAlMultiGPU(MadryEtAl):

    """
    A multi-GPU version of the Projected Gradient Descent Attack
    (Madry et al. 2016).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf

    This attack is designed to run on n-1 GPUs for generating adversarial
    examples and the last GPU will be used for doing the training step.
    Comparing to data parallelism, using this parallelization we can get
    very close to optimal n times speed up using n GPUs. The current
    implementation gets close to 6x speed up on 8 GPUs.
    """

    def __init__(self, **kwargs):
        """
        Create a MadryEtAlMultiGPU instance.
        """
        super(MadryEtAlMultiGPU, self).__init__(**kwargs)

    def attack(self, x, **kwargs):
        device_name = '/gpu:0'
        self.model.set_device(device_name)
        with tf.device(device_name):
            with tf.variable_scope('init_rand'):
                eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
                eta = self.clip_eta(eta)
                eta = tf.stop_gradient(eta)

                if self.y is not None:
                    y = self.y
                else:
                    preds = self.model.get_probs(x)
                    preds_max = tf.reduce_max(preds, 1, keep_dims=True)
                    y = tf.to_float(tf.equal(preds, preds_max))
                    y = y / tf.reduce_sum(y, 1, keep_dims=True)
                y = tf.stop_gradient(y)

        inputs = []
        outputs = []

        for i in range(self.nb_iter):
            # need at least 2 gpus
            gid = i % (self.ngpu-1)
            device_name = '/gpu:%d' % gid
            self.model.set_device(device_name)
            with tf.device(device_name):
                with tf.variable_scope('step%d' % i):
                    if i == 0:
                        inputs += [()]  # (x_pre, y)
                    else:
                        x = clone_variable('x', x)
                        eta = clone_variable('eta', eta)
                        y = clone_variable('y', y)
                        inputs += [(x, eta, y)]

                    x, eta = self.attack_single_step(x, eta, y)

                    if i < self.nb_iter-1:
                        outputs += [(x, eta, y)]
                    else:
                        adv_x = x + eta
                        if (self.clip_min is not None
                                and self.clip_max is not None):
                            adv_x = tf.clip_by_value(adv_x, self.clip_min,
                                                     self.clip_max)
                        adv_x = tf.stop_gradient(adv_x)
                        outputs += [(x, adv_x)]

        return inputs, outputs

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param ngpu: (required int) the number of GPUs available. ngpu-1
                     will be used in this attack.
        :param kwargs: A dictionary of parameters for MadryEtAl attack.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        adv_x = self.attack(x)

        return adv_x

    def parse_params(self, ngpu=1, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param ngpu: (required int) the number of GPUs available. ngpu-1
                     will be used in this attack.
        :param kwargs: A dictionary of parameters for MadryEtAl attack.
        """

        return_status = super(MadryEtAlMultiGPU, self).parse_params(**kwargs)
        self.ngpu = ngpu

        if self.ngpu < 2:
            raise ValueError("At least 2 GPUs need to be available.")

        return return_status
