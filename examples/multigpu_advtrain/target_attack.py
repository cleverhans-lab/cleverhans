import numpy as np
import tensorflow as tf

from cleverhans.model import Model, CallableModelWrapper
from cleverhans.attacks import Attack, FastGradientMethod
from cleverhans.utils_tf import model_loss

from model import clone_variable, MLP_probs


class ProjectedGradientDescentMethod(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a ProjectedGradientDescentMethod instance.
        """
        super(ProjectedGradientDescentMethod, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                'eps_iter': np.float32,
                                'y': np.float32,
                                'y_target': np.float32,
                                'clip_min': np.float32,
                                'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
            self.model = MLP_probs(self.model)

    def clip_eta(self, eta):
        # Clipping perturbation eta to self.ord norm ball
        if self.ord == np.inf:
            eta = tf.clip_by_value(eta, -self.eps, self.eps)
        elif self.ord in [1, 2]:
            reduc_ind = list(xrange(1, len(eta.get_shape())))
            if self.ord == 1:
                norm = tf.reduce_sum(tf.abs(eta),
                                     reduction_indices=reduc_ind,
                                     keep_dims=True)
            elif self.ord == 2:
                norm = tf.sqrt(tf.reduce_sum(tf.square(eta),
                                             reduction_indices=reduc_ind,
                                             keep_dims=True))
            eta = eta * self.eps / norm
        return eta

    def PGDpgd(self, x, **kwargs):
        from cleverhans.utils_tf import model_loss

        eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
        eta = self.clip_eta(eta)

        if self.y is not None:
            y = self.y
        else:
            preds = self.model.get_probs(x)
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            y = tf.to_float(tf.equal(preds, preds_max))
            y = y / tf.reduce_sum(y, 1, keep_dims=True)
        y = tf.stop_gradient(y)

        for i in range(self.nb_iter):
            adv_x = tf.stop_gradient(x + eta)
            preds = self.model.get_probs(adv_x)
            loss = model_loss(y, preds)
            grad, = tf.gradients(loss, adv_x)
            scaled_signed_grad = self.eps_iter * tf.sign(grad)
            adv_x = adv_x + scaled_signed_grad
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
            eta = adv_x - x
            eta = self.clip_eta(eta)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        adv_x = tf.stop_gradient(adv_x)

        return adv_x

    def PGDpgd_singleStep(self, x, eta, y):
        adv_x = tf.stop_gradient(x + eta)
        preds = self.model.get_probs(adv_x)
        loss = model_loss(y, preds)
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = self.clip_eta(eta)
        return x, eta, y

    def PGDpgd_multigpu(self, x, **kwargs):
        assert self.ngpu > 1

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

                    x, eta, y = self.PGDpgd_singleStep(x, eta, y)

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

    def PGDcl(self, x, **kwargs):
        eta = tf.truncated_normal(tf.shape(x), self.eps / 2)
        eta = self.clip_eta(eta)
        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y is not None:
            y = self.y
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
        y = tf.stop_gradient(y)

        fgm_params = {'eps': self.eps_iter, 'y': y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}
        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            eta = FGM.generate(x + eta, **fgm_params) - x

            eta = self.clip_eta(eta)

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        if self.pgd:
            if self.multigpu:
                adv_x = self.PGDpgd_multigpu(x)
            else:
                adv_x = self.PGDpgd(x)
        else:
            adv_x = self.PGDcl(x)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, pgd=True, multigpu=False, ngpu=1,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.pgd = pgd
        self.multigpu = multigpu
        self.ngpu = ngpu

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th':
            error_string = ("ProjectedGradientDescentMethod is"
                            " not implemented in Theano")
            raise NotImplementedError(error_string)

        return True
