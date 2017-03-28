from abc import ABCMeta
import numpy as np
import warnings

from .utils import random_targets


class Attack:
    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, pred, back='tf', sess=None, params={}):
        """
        :param pred: The model's symbolic output.
        :param back: The backend to use. Either 'tf' (default) or 'th'.
        :param params: Parameter dictionary used by child classes.
        """
        if not back == 'tf' or back == 'th':
            raise Exception("Backend argument must either be 'tf' or 'th'.")
        if back == 'tf' and sess is None:
            raise Exception("A tf session was not provided in sess argument.")
        if back == 'th' and sess is not None:
            raise Exception("A session should not be provided when using th.")

        # Prepare attributes
        self.x = None
        self.pred = pred
        self.back = back
        self.sess = sess
        self.nb_calls_generate = 0
        self.default_graph = None

    def generate(self, x, params={}):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overriden in any child class that implements an
        attack that is expressable symbolically.
        :param x: The model's symbolic inputs.
        :param params: Parameter dictionary used by child classes.
        :return: A symbolic representation of the adversarial examples.
        """
        # Keep track of the number of calls to warn when more than one default
        # graph to run if generating adversarial examples numerically
        self.nb_calls_generate += 1

        self.x = x

        if self.back == 'tf':
            import tensorflow as tf
            wrapper = tf.py_func(self.generate_np, [self.x], tf.float32)
            if self.nb_calls_generate == 1:
                self.default_graph = wrapper
            return wrapper
        else:
            # TODO()
            return False

    def generate_np(self, X, params={}):
        """
        Generate adversarial examples and return them as a Numpy array. This
        method should be overriden in any child class that implements an attack
        that is not fully expressed symbolically.
        :param X: A Numpy array with the original inputs.
        :param params: Parameter dictionary used by child classes.
        :return: A Numpy array holding the adversarial examples.
        """
        if self.default_graph is None:
            error_string = "The attack symbolic graph was not generated."
            raise NotImplementedError(error_string)
        if self.nb_calls_generate > 1:
            warnings.warn("Attack was generated symbolically multiple "
                          "times, using graph defined by first call.")

        if self.back == 'tf':
            from .utils_tf import batch_eval
            eval_params = {'batch_size': 128}
            X_adv, = batch_eval(self.sess, [self.x], [self.default_graph], [X],
                                args=eval_params)
            return X_adv
        else:
            # TODO()
            return False


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

        # Check that all required attack specific parameters are defined
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
        # TODO(This will not work with Theano because of sess)
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
    def __init__(self, x, pred, back='tf', sess=None, clip_min=None,
                 clip_max=None, params={'eps': 0.3,
                                        'eps_iter': 0.05,
                                        'ord': np.inf,
                                        'nb_iter': 10,
                                        'y': None}):
        """
        Create a BasicIterativeMethod instance.

        Attack-specific parameters:
        :param eps: (required float) maximum allowed perturbation distance
                    per feature. TODO(should this be per feature or per input?)
        :param eps_iter: (required float) step size for each attack iteration
        :param ord: (required) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param nb_iter: (required int) The number of iterations to run.
        :param y: (required) A placeholder for the model labels.
        """
        super(BasicIterativeMethod, self).__init__(x, pred, back, sess,
                                                   clip_min, clip_max, params)

        # Check that all required attack specific parameters are defined
        required_params = ('eps', 'eps_iter', 'ord', 'nb_iter', 'y')
        assert all(k in params for k in required_params)
        assert params['y'] is not None, "Attack requires label placeholder."

        # Check if order of the norm is acceptable given current implementation
        if params['ord'] not in [np.inf, 1, 2]:
            raise Exception("Norm order must be either np.inf, 1, or 2.")
        if back == 'th' and params['ord'] != np.inf:
            raise NotImplementedError("The only BasicIterativeMethod norm "
                                      "implemented for Theano is np.inf.")

        # Save attack-specific parameters
        self.eps = params['eps']
        self.eps_iter = params['eps_iter']
        self.nb_iter = params['nb_iter']
        self.y = params['y']

        # Initialize symbolic graph of FastGradientMethod used in iterations
        self.fgm = FastGradientMethod(x, pred, back, sess, clip_min, clip_max,
                                      params={'eps': params['eps_iter'],
                                              'ord': params['ord'],
                                              'y': self.y})
        self.fgm.generate_symbolic()

    def craft(self, X, params={'Y': None, 'batch_size': 128}):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        super(BasicIterativeMethod, self).craft(X, params)

        # Verify label placeholder was defined previously if using true labels
        if params['Y'] is not None:
            error = "True labels given but label placeholder missing in _init_"
            assert self.y is not None, error

        # Define clipping extrema
        upper_bound = X + self.eps
        lower_bound = X - self.eps

        # Iteratively apply the FastGradientMethod
        X_adv = X
        for i in range(self.nb_iter):
            # TODO(This implementation is wrong because the label should be)
            # TODO(fixed after the first iteration. This attack can be)
            # TODO(defined symbolically using a tf loop.)
            X_adv = self.fgm.craft(X_adv, params)
            X_adv = np.clip(X_adv, a_min=lower_bound, a_max=upper_bound)

        return X_adv


class SaliencyMapMethod(Attack):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """
    def __init__(self, x, pred, back='tf', sess=None, clip_min=None,
                 clip_max=None, params={'theta': 1.,
                                        'gamma': np.inf,
                                        'nb_classes': 2}):
        """
        Create a SaliencyMapMethod instance.

        Attack-specific parameters:
        :param theta: (required float) perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (required float) Maximum percentage of perturbed features
        :param nb_classes: (required int) Number of model output classes
        """
        super(SaliencyMapMethod, self).__init__(x, pred, back, sess,
                                                clip_min, clip_max,
                                                params)

        # Check that all required attack specific parameters are defined
        required_params = ('theta', 'gamma', 'nb_classes')
        assert all(k in params for k in required_params)

        self.theta = params['theta']
        self.gamma = params['gamma']
        self.nb_classes = params['nb_classes']

        if self.back == 'tf':
            from .attacks_tf import jacobian_graph
        else:
            raise NotImplementedError('Theano version of SaliencyMapMethod not'
                                      ' currently implemented.')

        self.grads = jacobian_graph(pred, x, params['nb_classes'])

    def craft(self, X, params={'Y': None,
                               'batch_size': 128,
                               'targets': None}):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        super(SaliencyMapMethod, self).craft(X, params)

        # If targets were specified, make sure we have as many as the inputs
        if params['targets'] is not None:
            if len(params['targets'].shape) > 1:
                nb_targets = len(params['targets'])
            else:
                nb_targets = 1
        if params['targets'] is not None and nb_targets != len(X):
            raise Exception("Must specify exactly one target per input.")

        X_adv = np.zeros(X.shape)

        # TODO(Optimize underlying functions to remove this loop: issue #)
        for ind, val in enumerate(X):
            val = np.expand_dims(val, axis=0)
            if params['targets'] is None:
                # No targets provided, randomly choose from incorrect classes
                if params['Y'] is None:
                    # No true labels given: use model pred as ground truth
                    from .utils_tf import model_argmax
                    gt = model_argmax(self.sess, self.x, self.pred, val)
                else:
                    # True labels were provided
                    gt = np.argmax(params['Y'][ind], axis=1)

                # Randomly choose from the incorrect classes for each sample
                target = random_targets(gt, self.nb_classes)[0]
            else:
                target = params['targets'][ind]

            from .attacks_tf import jsma
            X_adv[ind], _, _ = jsma(self.sess, self.x, self.pred, self.grads,
                                    val, np.argmax(target), self.theta,
                                    self.gamma, self.clip_min, self.clip_max)

        return X_adv


def fgsm(x, predictions, eps, back='tf', clip_min=None, clip_max=None):
    """
    A wrapper for the Fast Gradient Sign Method.
    It calls the right function, depending on the
    user's backend.
    :param x: the input
    :param predictions: the model's output
                        (Note: in the original paper that introduced this
                         attack, the loss was computed by comparing the
                         model predictions with the hard labels (from the
                         dataset). Instead, this version implements the loss
                         by comparing the model predictions with the most
                         likely class. This tweak is recommended since the
                         discovery of label leaking in the following paper:
                         https://arxiv.org/abs/1611.01236)
    :param eps: the epsilon (input variation parameter)
    :param back: switch between TensorFlow ('tf') and
                Theano ('th') implementation
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """
    warnings.warn("attacks.fgsm is deprecated and will be removed on 09-27-17."
                  " Instantiate an attack object from FastGradientMethod.")
    if back == 'tf':
        # Compute FGSM using TensorFlow
        from .attacks_tf import fgm
        return fgm(x, predictions, y=None, eps=eps, ord=np.inf,
                   clip_min=clip_min, clip_max=clip_max)
    elif back == 'th':
        # Compute FGSM using Theano
        from .attacks_th import fgm
        return fgm(x, predictions, eps, clip_min=clip_min, clip_max=clip_max)


def jsma(sess, x, predictions, grads, sample, target, theta, gamma=np.inf,
         increase=True, back='tf', clip_min=None, clip_max=None):
    """
    A wrapper for the Jacobian-based saliency map approach.
    It calls the right function, depending on the
    user's backend.
    :param sess: TF session
    :param x: the input
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param sample: (1 x 1 x img_rows x img_cols) numpy array with sample input
    :param target: target class for input sample
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param back: switch between TensorFlow ('tf') and
                Theano ('th') implementation
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: an adversarial sample
    """
    warnings.warn("attacks.jsma is deprecated and will be removed on 09-27-17."
                  " Instantiate an attack object from SaliencyMapMethod.")
    if back == 'tf':
        # Compute Jacobian-based saliency map attack using TensorFlow
        from .attacks_tf import jsma
        return jsma(sess, x, predictions, grads, sample, target, theta, gamma,
                    clip_min, clip_max)
    elif back == 'th':
        raise NotImplementedError("Theano jsma not implemented.")
