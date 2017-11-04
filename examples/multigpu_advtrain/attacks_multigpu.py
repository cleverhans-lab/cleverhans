import tensorflow as tf

from cleverhans.attacks import MadryEtAl
from cleverhans.utils_tf import clip_eta

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

    def __init__(self, *args, **kwargs):
        """
        Create a MadryEtAlMultiGPU instance.
        """
        super(MadryEtAlMultiGPU, self).__init__(*args, **kwargs)
        self.structural_kwargs += ['ngpu', 'g0_inputs']

    def get_or_guess_labels(self, x, kwargs):
        device_name = '/gpu:0'
        self.model.set_device(device_name)
        with tf.device(device_name):
            with tf.variable_scope('model_pred'):
                ret = super(MadryEtAlMultiGPU, self).get_or_guess_labels(
                    x, kwargs)
        return ret

    def attack(self, x, y, **kwargs):
        """
        This method creates a symoblic graph of the MadryEtAl attack on
        multiple GPUs. The assumption is that at least 2 GPUs exist. The graph
        is created on the first n-1 GPUs. The last GPU is left for train step.

        Stop gradient is needed to get the speed-up. This prevents us from
        being able to back-prop through the attack.

        :param x: A tensor with the input image.
        :return: Two lists containing the input and output tensors of each GPU.
        """
        # Create the initial random perturbation
        device_name = '/gpu:0'
        self.model.set_device(device_name)
        with tf.device(device_name):
            with tf.variable_scope('init_rand'):
                eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
                eta = clip_eta(eta, self.ord, self.eps)
                eta = tf.stop_gradient(eta)

        # List of inputs/outputs for each GPU
        inputs = []
        outputs = []

        for i in range(self.nb_iter):
            # Create the graph for i'th step of attack
            # Last GPU is reserved for training step
            gid = i % (self.ngpu-1)
            device_name = '/gpu:%d' % gid
            self.model.set_device(device_name)
            with tf.device(device_name):
                with tf.variable_scope('step%d' % i):
                    if i == 0:
                        # This will be filled by the TrainerMultiGPU
                        x_pre, _, y0 = self.g0_inputs
                        inputs += [(x_pre, y0)]
                    else:
                        # Clone the variables to separate the graph of 2 GPUs
                        x = clone_variable('x', x)
                        eta = clone_variable('eta', eta)
                        y = clone_variable('y', y)
                        # copy y0 forward
                        y0 = clone_variable('y0', self.g0_inputs[2])
                        inputs += [(x, eta, y, y0)]

                    x, eta = self.attack_single_step(x, eta, y)

                    if i < self.nb_iter-1:
                        outputs += [(x, eta, y, y0)]
                    else:
                        # adv_x, not eta is the output of attack
                        # No need to output y anymore. It was used only inside
                        # this attack
                        adv_x = x + eta
                        if (self.clip_min is not None
                                and self.clip_max is not None):
                            adv_x = tf.clip_by_value(adv_x, self.clip_min,
                                                     self.clip_max)
                        adv_x = tf.stop_gradient(adv_x)
                        outputs += [(x, adv_x, y0)]

        return inputs, outputs

    def parse_params(self, ngpu=1, g0_inputs=None, **kwargs):
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
        self.g0_inputs = g0_inputs

        if self.ngpu < 2:
            raise ValueError("At least 2 GPUs need to be available.")
        if len(g0_inputs) != 3:
            raise ValueError("g0_inputs should be a tuple of 3 (x_pre, x, y).")

        return return_status
