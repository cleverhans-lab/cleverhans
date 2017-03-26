from abc import ABCMeta, abstractmethod
import numpy as np
import warnings

from .utils import random_targets


class Attack:
    """
    Abstract base class for all attack classes. All attacks must override the
    `craft` method that returns adversarial examples corresponding to the input
    data. However, the `generate_symbolic` method is optional.
    """
    __metaclass__ = ABCMeta

    def __init__(self, x, pred, back='tf', sess=None, clip_min=None,
                 clip_max=None, params={}):
        """
        :param x: The model's symbolic inputs.
        :param pred: The model's symbolic output.
        :param back: The backend to use. Either 'tf' (default) or 'th'.
        :param clip_min: Minimum float value for adversarial example components
        :param clip_max: Maximum float value for adversarial example components
        :param params: Parameter dictionary used by child classes.
        """
        if not back == 'tf' or back == 'th':
            raise Exception("Backend argument must either be 'tf' or 'th'.")
        if back == 'tf' and sess is None:
            raise Exception("A tf session was not provided in sess argument.")
        if back == 'th' and sess is not None:
            raise Exception("A session should not be provided when using th.")
        self.x = x
        self.pred = pred
        self.back = back
        self.sess = sess
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate_symbolic(self):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overwritten in any child class that implements an
        attack that is expressable symbolically.
        :return: A symbolic representation of the adversarial examples.
        """
        error_string = "This attack is not (yet) implemented symbolically."
        raise NotImplementedError(error_string)

    @abstractmethod
    def craft(self, X, params={}):
        """
        Generate adversarial examples and return them as a Numpy array.
        :param X: A Numpy array with the original inputs.
        :param params: Parameter dictionary used by child classes.
        :return: A Numpy array holding the adversarial examples.
        """


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, x, pred, back='tf', sess=None, clip_min=None,
                 clip_max=None, params={'eps': 0.3,
                                        'ord': 'np.inf',
                                        'y': None}):
        """
        Create a FastGradientMethod instance.

        Attack-specific parameters:
        :param eps: (required float) attack step size (input variation)
        :param ord: (required) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the label leaking effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        """
        super(FastGradientMethod, self).__init__(x, pred, back, sess, clip_min,
                                                 clip_max, params)
        assert 'eps' in params and 'ord' in params

        # Check if order of the norm is acceptable given current implementation
        if params['ord'] not in [np.inf, int(1), int(2)]:
            raise Exception("Norm order must be either np.inf, 1, or 2.")
        if back == 'th' and params['ord'] != np.inf:
            raise NotImplementedError("The only FastGradientMethod norm "
                                      "implemented for Theano is np.inf.")

        # Save attack-specific parameters
        self.eps = params['eps']
        self.ord = params['ord']
        if 'y' in params:
            self.y = params['y']
        else:
            self.y = None

        # Create symbolic graph defining adversarial examples
        self.x_adv = self.generate_symbolic()

    def generate_symbolic(self):
        """
        Generate symbolic graph for adversarial examples and return.
        """
        if self.back == 'tf':
            from .attacks_tf import fgm as fgsm
        else:
            from .attacks_th import fgsm

        return fgsm(self.x, self.pred, self.y, self.eps, self.ord,
                    self.clip_min, self.clip_max)

    def craft(self, X, params={'Y': None, 'batch_size': 128}):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        super(FastGradientMethod, self).craft(X, params)

        # Verify label placeholder was defined previously if using true labels
        if params['Y'] is not None:
            error = "True labels given but label placeholder missing in _init_"
            assert self.y is not None, error

        # Define appropriate batch_eval function for chosen backend
        if self.back == 'tf':
            from .utils_tf import batch_eval
        else:
            from .utils_th import batch_eval
        eval_params = {'batch_size': params['batch_size']}

        # Run symbolic graph without or with true labels
        if params['Y'] is None:
            X_adv, = batch_eval(self.sess, [self.x], [self.x_adv], [X],
                                args=eval_params)
        else:
            X_adv, = batch_eval(self.sess, [self.x, self.y], [self.x_adv],
                                [X, params['Y']], args=eval_params)

        return X_adv


class BasicIterativeMethod(Attack):
    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """
    def __init__(self, x, pred, y=None, back='tf', clip_min=None,
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
        super(BasicIterativeMethod, self).__init__(x, pred, y, back,
                                                   clip_min, clip_max,
                                                   other_params)
        if other_params['ord'] not in ['inf', 'L1', 'L2']:
            raise Exception("'ord' param must be either 'inf', 'L1', or 'L2'.")
        if back == 'th' and other_params['ord'] != 'inf':
            raise NotImplementedError("The only BasicIterativeMethod norm currently "
                                      "implemented for Theano is 'inf'.")
        self.eps = other_params['eps']
        self.eps_iter = other_params['eps_iter']
        self.nb_iter = other_params['nb_iter']
        self.fgm = FastGradientMethod(
            x, pred, y, back, clip_min, clip_max,
            other_params={'eps': other_params['eps_iter'],
                          'ord': other_params['ord']}
        )

    def craft(self, X, Y=None, sess=None, batch_size=128, target=None):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        super(BasicIterativeMethod, self).craft(X, Y, sess, batch_size, target)
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
            X_adv = self.fgm.craft(X_adv, Y, sess, batch_size)
            X_adv = np.minimum(np.maximum(X_adv, lower_bound), upper_bound)

        return X_adv


class SaliencyMapMethod(Attack):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """
    def __init__(self, x, pred, y=None, back='tf', clip_min=None,
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
        super(SaliencyMapMethod, self).__init__(x, pred, y, back,
                                                clip_min, clip_max,
                                                other_params)
        self.theta = other_params['theta']
        self.gamma = other_params['gamma']
        self.increase = other_params['increase']
        self.nb_classes = other_params['nb_classes']
        if self.back == 'tf':
            from .attacks_tf import jacobian_graph
        else:
            raise NotImplementedError('Theano version of SaliencyMapMethod not'
                                      ' currently implemented.')
        self.grads = jacobian_graph(pred, x, other_params['nb_classes'])

    def craft(self, X, Y=None, sess=None, batch_size=128, target=None):
        """
        Generate adversarial samples and return them in a Numpy array.
        NOTE: this attack currently only computes one sample at a time.
        """
        super(SaliencyMapMethod, self).craft(X, Y, sess, batch_size, target)
        if len(X) > 1:
            raise Exception('SaliencyMapMethod currently only handles one sample'
                            'at a time. Make sure that len(X) = 1.')
        if target is None:
            # No targets provided, so we will randomly choose targets from the
            # incorrect classes
            if Y is None:
                # No true labels provided: use model predictions as ground truth
                if self.back == 'tf':
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
        if self.back == 'tf':
            from .attacks_tf import jsma
        else:
            # no need to raise notimplemented error again; should have been
            # done during initialization
            pass
        X_adv, _, _ = jsma(sess, self.x, self.pred, self.grads, X, target,
                           self.theta, self.gamma, self.increase,
                           self.clip_min, self.clip_max)

        return X_adv
