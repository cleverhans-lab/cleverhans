from abc import ABCMeta, abstractmethod
import numpy as np
import warnings

from .utils import random_targets


class Attack:
    """
    Abstract base class for all attack classes. All attacks must override the
    generate_numpy method that returns the adversarial examples corresponding
    to the input data. However, the generate_symbolic method is optional.
    """
    __metaclass__ = ABCMeta

    def __init__(self, x, pred, y=None, backend='tf',
                 clip_min=None, clip_max=None, other_params={}):
        """
        :param x: A placeholder for the model inputs.
        :param pred: The model's symbolic output.
        :param y: A placeholder for the model labels. Only provide this
                  parameter if you'd like to use the true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the label leaking effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param backend: A string indicating which backend to use. Must be
                        either 'tf' or 'th'. Default is 'tf'.
        :param clip_min: Optional float parameter that can be used to set a
                         minimum value for components of the example returned.
        :param clip_max: Optional float parameter that can be used to set a
                         maximum value for components of the example returned.
        :param other_params: Dictionary holding the other parameters needed for
                            various child classes
        """
        if not backend == 'tf' or backend == 'th':
            raise Exception("Backend argument must be 'tf' or 'th'.")
        self.x = x
        self.pred = pred
        self.y = y
        self.backend = backend
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate_symbolic(self):
        """
        Generate symbolic graph for adversarial samples and return. This
        method should be overwritten in any child class which has a
        symbolic implementation available.
        :return: A symbolic representation of the adversarial samples.
        """
        raise NotImplementedError("The symbolic version of this attack "
                                  "is not currently implemented.")

    @abstractmethod
    def generate_numpy(self, X, Y=None, sess=None, batch_size=128, target=None):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param X: A Numpy array representing the feature matrix for the baseline
                samples.
        :param Y: A Numpy array representing the label matrix for the baseline
                samples. Default is None (this is only used when using true labels
                to craft adversarial samples). Labels should be one-hot-encoded.
        :param sess: A TensorFlow session to use for evaluating the adversarial
                    samples (for 'tf' backend only). Default is None.
        :param batch_size: An int indicating the batch size to use when evaluating
                        adversarial samples.
        :param target: A numpy array holding the target classes for each
                    adversarial sample.
        :return: A Numpy array holding the adversarial samples.
        """
        if self.backend == 'tf':
            # Using TensorFlow backend; must pass a session argument.
            assert sess is not None, 'tf session must be provided when using TensorFlow backend.'
        else:
            # Using Theano backend; no session required.
            assert sess is None, 'session argument was provided, but we are using Theano backend.'


class FastGradientMethod(Attack):
    """
    The Fast Gradient Method. This attack was originally implemented by
    Goodfellow et al. (2015) with the infinity norm ("Fast Gradient
    Sign Method").
    Paper link: https://arxiv.org/pdf/1412.6572.pdf
    """
    def __init__(self, x, pred, y=None, backend='tf', clip_min=None,
                 clip_max=None, other_params={'eps': 0.3, 'ord': 'inf'}):
        """
        Create a FastGradientMethod instance.

        Attack-specific parameters:
        :param eps: A float indicating the step size to use for the adversarial
                    algorithm (input variation parameter).
        :param ord: A string indicating the norm order to use when computing
                    gradients. This should be either 'inf', 'L1' or 'L2'.
        """
        super(FastGradientMethod, self).__init__(x, pred, y, backend,
                                                 clip_min, clip_max,
                                                 other_params)
        if other_params['ord'] not in ['inf', 'L1', 'L2']:
            raise Exception("'ord' param must be either 'inf', 'L1', or 'L2'.")
        if backend == 'th' and other_params['ord'] != 'inf':
            raise NotImplementedError("The only FastGradientMethod norm "
                                      "implemented for Theano is 'inf'.")
        self.eps = other_params['eps']
        self.ord = other_params['ord']
        # create symbolic adversarial sample graph
        self.x_adv = self.generate_symbolic()

    def generate_symbolic(self):
        """
        Generate symbolic graph for adversarial samples and return.
        """
        if self.backend == 'tf':
            from .attacks_tf import fgsm
        else:
            from .attacks_th import fgsm

        return fgsm(self.x, self.pred, self.y, self.eps, self.ord,
                    self.clip_min, self.clip_max)

    def generate_numpy(self, X, Y=None, sess=None, batch_size=128, target=None):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        super(FastGradientMethod, self).generate_numpy(X, Y, sess, batch_size, target)
        # targets not currently used by this attack
        if target is not None:
            warnings.warn("Ignoring 'target' argument: the use of targets is not "
                          "currently implemented for this attack.")
        # verify we are only using true labels if we indicated so previously
        if Y is not None:
            assert self.y is not None, 'Label placeholder must be provided to _init_ ' \
                                       'in order to use true labels'
        if self.backend == 'tf':
            # Tensorflow backend; evaluate symbolic samples
            from .utils_tf import batch_eval
        else:
            # Theano backend; evaluate symbolic samples
            from .utils_th import batch_eval
        eval_params = {'batch_size': batch_size}
        if Y is not None:
            X_adv, = batch_eval(sess, [self.x, self.y], [self.x_adv],
                                [X, Y], args=eval_params)
        else:
            X_adv, = batch_eval(sess, [self.x], [self.x_adv], [X],
                                args=eval_params)

        return X_adv


class BasicIterativeMethod(Attack):
    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """
    def __init__(self, x, pred, y=None, backend='tf', clip_min=None,
                 clip_max=None, other_params={'eps': 0.3, 'eps_iter': 0.05,
                                              'ord': 'inf', 'nb_iter': 10}):
        """
        Create a BasicIterativeMethod instance.

        Attack-specific parameters:
        :param eps: A float indicating the maximum allowed perturbation
                    distance for each feature.
        :param eps_iter: A float indicating the step size to use for each
                        iteration of BIM (input variation parameter).
        :param ord: A string indicating the norm order to use when computing
                    gradients. This should be either 'inf', 'L1' or 'L2'.
        :param nb_iter: The number of BIM iterations to run.
        """
        super(BasicIterativeMethod, self).__init__(x, pred, y, backend,
                                                   clip_min, clip_max,
                                                   other_params)
        if other_params['ord'] not in ['inf', 'L1', 'L2']:
            raise Exception("'ord' param must be either 'inf', 'L1', or 'L2'.")
        if backend == 'th' and other_params['ord'] != 'inf':
            raise NotImplementedError("The only BasicIterativeMethod norm currently "
                                      "implemented for Theano is 'inf'.")
        self.eps = other_params['eps']
        self.eps_iter = other_params['eps_iter']
        self.nb_iter = other_params['nb_iter']
        self.fgm = FastGradientMethod(
            x, pred, y, backend, clip_min, clip_max,
            other_params={'eps': other_params['eps_iter'],
                          'ord': other_params['ord']}
        )

    def generate_numpy(self, X, Y=None, sess=None, batch_size=128, target=None):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        super(BasicIterativeMethod, self).generate_numpy(X, Y, sess, batch_size, target)
        # targets not currently used by this attack
        if target is not None:
            warnings.warn("Ignoring 'target' argument: the use of targets is not "
                          "currently implemented for this attack.")
        # verify we are only using true labels if we indicated so previously
        if Y is not None:
            assert self.y is not None, 'Label placeholder must be provided to _init_ ' \
                                       'in order to use true labels'
        upper_bound = X + self.eps
        lower_bound = X - self.eps
        X_adv = X
        for i in range(self.nb_iter):
            X_adv = self.fgm.generate_numpy(X_adv, Y, sess, batch_size)
            X_adv = np.minimum(np.maximum(X_adv, lower_bound), upper_bound)

        return X_adv


class SaliencyMapMethod(Attack):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """
    def __init__(self, x, pred, y=None, backend='tf', clip_min=None,
                 clip_max=None, other_params={'theta': 1., 'gamma': np.inf,
                                              'increase': True,
                                              'nb_classes': 2}):
        """
        Create a SaliencyMapMethod instance.

        Attack-specific parameters:
        :param theta: A float indicating the delta for each feature adjustment.
        :param gamma: A float between 0 - 1 indicating the maximum distortion
                    percentage.
        :param increase: A boolean; True if we are increasing feature values,
                        False if we are decreasing.
        :param nb_classes: An integer specifying the number of classes in
                        this classification model.
        """
        super(SaliencyMapMethod, self).__init__(x, pred, y, backend,
                                                clip_min, clip_max,
                                                other_params)
        self.theta = other_params['theta']
        self.gamma = other_params['gamma']
        self.increase = other_params['increase']
        self.nb_classes = other_params['nb_classes']
        if self.backend == 'tf':
            from .attacks_tf import jacobian_graph
        else:
            raise NotImplementedError('Theano version of SaliencyMapMethod not'
                                      ' currently implemented.')
        self.grads = jacobian_graph(pred, x, other_params['nb_classes'])

    def generate_numpy(self, X, Y=None, sess=None, batch_size=128, target=None):
        """
        Generate adversarial samples and return them in a Numpy array.
        NOTE: this attack currently only computes one sample at a time.
        """
        super(SaliencyMapMethod, self).generate_numpy(X, Y, sess, batch_size, target)
        if len(X) > 1:
            raise Exception('SaliencyMapMethod currently only handles one sample'
                            'at a time. Make sure that len(X) = 1.')
        if target is None:
            # No targets provided, so we will randomly choose targets from the
            # incorrect classes
            if Y is None:
                # No true labels provided: use model predictions as ground truth
                if self.backend == 'tf':
                    from .utils_tf import model_argmax
                else:
                    from .utils_th import model_argmax
                gt = model_argmax(self.x, self.pred, X)
            else:
                # True labels were provided
                gt = np.argmax(Y, axis=1)
            # Randomly choose from the incorrect classes for each sample
            # TODO: remove [0] once we fix SaliencyMapMethod to handle multiple samples
            target = random_targets(gt, self.nb_classes)[0]
        else:
            if Y is not None:
                warnings.warn("Ignoring 'Y' argument since class targets were provided.")
        if self.backend == 'tf':
            from .attacks_tf import jsma
        else:
            # no need to raise notimplemented error again; should have been
            # done during initialization
            pass
        X_adv, _, _ = jsma(sess, self.x, self.pred, self.grads, X, target,
                           self.theta, self.gamma, self.increase,
                           self.clip_min, self.clip_max)

        return X_adv
