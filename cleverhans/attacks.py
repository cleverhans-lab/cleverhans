from abc import ABCMeta
import numpy as np
import warnings


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
        if not(back == 'tf' or back == 'th'):
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

    def generate(self, x):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overriden in any child class that implements an
        attack that is expressable symbolically. Otherwise, it will wrap the
        numerical implementation as a symbolic operator.
        :param x: The model's symbolic inputs.
        :return: A symbolic representation of the adversarial examples.
        """
        # Keep track of the number of calls to warn when more than one default
        # graph to run if generating adversarial examples numerically
        self.nb_calls_generate += 1

        if self.back == 'tf':
            import tensorflow as tf
            wrapper = tf.py_func(self.generate_np, [self.x], tf.float32)
            if self.nb_calls_generate == 1:
                self.x = x
                self.default_graph = wrapper
            return wrapper
        else:
            raise NotImplementedError('Theano version not implemented.')

    def generate_np(self, X, params={'batch_size': 128}):
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
            raise Exception(error_string)
        if self.nb_calls_generate > 1:
            warnings.warn("Attack was generated symbolically multiple "
                          "times, using graph defined by first call.")

        # Define batch_eval function common to both backends
        eval_params = {'batch_size': params['batch_size']}
        if self.back == 'tf':
            from .utils_tf import batch_eval

            def batch_eval_com(in_sym, out_sym, inputs):
                return batch_eval(self.sess, in_sym, out_sym, inputs,
                                  args=eval_params)
        else:
            from .utils_th import batch_eval

            def batch_eval_com(in_sym, out_sym, inputs):
                return batch_eval(in_sym, out_sym, inputs, args=eval_params)

        X_adv, = batch_eval_com([self.x], [self.default_graph], [X],
                                args=eval_params)
        return X_adv


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, pred, back='tf', sess=None, params={'eps': 0.3,
                                                           'ord': 'np.inf',
                                                           'y': None,
                                                           'clip_min': None,
                                                           'clip_max': None}):
        """
        Create a FastGradientMethod instance.

        Attack-specific parameters:
        :param eps: (required float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the label leaking effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        super(FastGradientMethod, self).__init__(pred, back, sess, params)

        # Check that all required attack specific parameters are defined
        assert 'eps' in params

        # Save attack-specific parameters
        self.eps = params['eps']
        self.ord = params['ord'] if 'ord' in params else np.inf
        self.y = params['y'] if 'y' in params else None
        self.clip_min = params['clip_min'] if 'clip_min' in params else None
        self.clip_max = params['clip_max'] if 'clip_max' in params else None

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise Exception("Norm order must be either np.inf, 1, or 2.")
        if back == 'th' and self.ord != np.inf:
            raise NotImplementedError("The only FastGradientMethod norm "
                                      "implemented for Theano is np.inf.")

    def generate(self, x):
        """
        Generate symbolic graph for adversarial examples and return.
        """
        self.nb_calls_generate += 1
        if self.nb_calls_generate == 1:
            self.x = x

        if self.back == 'tf':
            from .attacks_tf import fgm
        else:
            from .attacks_th import fgm

        graph = fgm(x, self.pred, y=self.y, eps=self.eps, ord=self.ord,
                    clip_min=self.clip_min, clip_max=self.clip_max)

        if self.nb_calls_generate == 1:
            self.default_graph = graph
        return graph

    def generate_np(self, X, params={'Y': None, 'batch_size': 128}):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        if self.default_graph is None:
            error_string = "The attack symbolic graph was not generated."
            raise Exception(error_string)
        if self.nb_calls_generate > 1:
            warnings.warn("Attack was generated symbolically multiple "
                          "times, using graph defined by first call.")

        # Verify label placeholder was defined previously if using true labels
        if params['Y'] is not None:
            error = "True labels given but label placeholder missing in _init_"
            assert self.y is not None, error

        # Define batch_eval function common to both backends
        eval_params = {'batch_size': params['batch_size']}
        if self.back == 'tf':
            from .utils_tf import batch_eval

            def batch_eval_com(in_sym, out_sym, inputs):
                return batch_eval(self.sess, in_sym, out_sym, inputs,
                                  args=eval_params)
        else:
            from .utils_th import batch_eval

            def batch_eval_com(in_sym, out_sym, inputs):
                return batch_eval(in_sym, out_sym, inputs, args=eval_params)

        # Run symbolic graph without or with true labels
        if params['Y'] is None:
            X_adv, = batch_eval_com([self.x], [self.default_graph], [X])
        else:
            X_adv, = batch_eval_com([self.x, self.y], [self.default_graph],
                                    [X, params['Y']])

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
    def __init__(self, pred, back='tf', sess=None, params={'theta': 1.,
                                                           'gamma': np.inf,
                                                           'nb_classes': 10,
                                                           'clip_min': 0.,
                                                           'clip_max': 1.}):
        """
        Create a SaliencyMapMethod instance.

        Attack-specific parameters:
        :param theta: (required float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (required float) Maximum percentage of perturbed features
        :param nb_classes: (required int) Number of model output classes
        :param clip_min: (required float) Minimum component value for clipping
        :param clip_max: (required float) Maximum component value for clipping
        """
        super(SaliencyMapMethod, self).__init__(pred, back, sess, params)

        # Check that all required attack specific parameters are defined
        req = ('theta', 'gamma', 'nb_classes', 'clip_min', 'clip_max')
        if not all(key in params for key in req):
            raise Exception("JSMA must be instantiated w/ params: " + str(req))

        self.theta = params['theta']
        self.gamma = params['gamma']
        self.nb_classes = params['nb_classes']
        self.clip_min = params['clip_min']
        self.clip_max = params['clip_max']

        self.x = None
        self.pred = pred
        self.targeted = None  # Target placeholder if the attack is targeted

    def generate(self, x, params={'targets': None}):
        """
        Attack-specific parameters:
        :param targets: (optional) Target placeholder if the attack is targeted
        """
        # Give default value to undefined optional parameters
        if 'targets' not in params:
            params['targets'] = None

        if self.back == 'tf':
            self.nb_calls_generate += 1

            import tensorflow as tf
            from .attacks_tf import jacobian_graph, jsma_batch

            # Define Jacobian graph wrt to this input placeholder
            grads = jacobian_graph(self.pred, x, self.nb_classes)

            # Define appropriate graph (targeted / random target labels)
            if params['targets'] is not None:
                def jsma_wrap(X, targets):
                    return jsma_batch(self.sess, x, self.pred, grads, X,
                                      self.theta, self.gamma, self.clip_min,
                                      self.clip_max, self.nb_classes,
                                      targets=targets)

                # Attack is targeted, target placeholder will need to be fed
                jsma_wrap_args = [x, params['targets']]
                wrap = tf.py_func(jsma_wrap, jsma_wrap_args, tf.float32)
            else:
                def jsma_wrap(X):
                    return jsma_batch(self.sess, x, self.pred, grads, X,
                                      self.theta, self.gamma, self.clip_min,
                                      self.clip_max, self.nb_classes,
                                      targets=None)

                # Attack is untargeted, target values will be chosen at random
                wrap = tf.py_func(jsma_wrap, [x], tf.float32)

            if self.nb_calls_generate == 1:
                self.x = x
                self.default_graph = wrap
                self.targeted = params['targets']

            return wrap
        else:
            raise NotImplementedError('Theano version of SaliencyMapMethod not'
                                      ' currently implemented.')

    def generate_np(self, X, params={'batch_size': 128, 'targets': None}):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        if self.default_graph is None:
            error_string = "The attack symbolic graph was not generated."
            raise Exception(error_string)
        if self.nb_calls_generate > 1:
            warnings.warn("Attack was generated symbolically multiple "
                          "times, using graph defined by first call.")

        # Give default value to undefined optional parameters
        if 'batch_size' not in params:
            params['batch_size'] = 128
        if 'targets' not in params:
            params['targets'] = None

        # If targets were specified, make sure we have as many as the inputs
        # and that the graph was generated in a targeted way.
        if params['targets'] is not None:
            if len(params['targets'].shape) > 1:
                nb_targets = len(params['targets'])
            else:
                nb_targets = 1
            if nb_targets != len(X):
                raise Exception("Must specify exactly one target per input.")
            if self.targeted is None:
                raise Exception("Trying to run targeted attack on graph that"
                                " was generated un-targeted.")

        if self.back == 'tf':
            from .utils_tf import batch_eval
            eval_params = {'batch_size': params['batch_size']}

            # Run appropriate graph (with or without target labels)
            if params['targets'] is not None:
                X_adv, = batch_eval(self.sess, [self.x, self.targeted],
                                    [self.default_graph],
                                    [X, params['targets']], args=eval_params)
            else:
                X_adv, = batch_eval(self.sess, [self.x], [self.default_graph],
                                    [X], args=eval_params)
            return X_adv
        else:
            raise NotImplementedError('Theano version of SaliencyMapMethod not'
                                      ' currently implemented.')


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
