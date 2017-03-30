from abc import ABCMeta
import numpy as np
import warnings


class Attack:
    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None, params={}):
        """
        :param pred: A function that takes a symbolic input and returns the
                     symbolic output for the model's predictions.
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
        self.model = model
        self.back = back
        self.sess = sess

    def generate(self, x, params={}):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overriden in any child class that implements an
        attack that is expressable symbolically. Otherwise, it will wrap the
        numerical implementation as a symbolic operator.
        :param x: The model's symbolic inputs.
        :param params: Parameter dictionary used by child classes.
        :return: A symbolic representation of the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        import tensorflow as tf
        return tf.py_func(self.generate_np, [x], tf.float32)

    def generate_np(self, X, params={}):
        """
        Generate adversarial examples and return them as a Numpy array. This
        method should be overriden in any child class that implements an attack
        that is not fully expressed symbolically.
        :param X: A Numpy array with the original inputs.
        :param params: Parameter dictionary used by child classes.
        :return: A Numpy array holding the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        import tensorflow as tf

        # Generate this attack's graph if it hasn't been done previously
        if not hasattr(self, "_x") and not hasattr(self, "_x_adv"):
            input_shape = list(X.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate(self._x)

        # This indicates loop calls between generate and generate_np
        if hasattr(self, "_x") and not hasattr(self, "_x_adv"):
            error_string = "No symbolic or numeric implementation of attack."
            raise NotImplementedError(error_string)

        return self.sess.run(self._x_adv, feed_dict={self._x: X})


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, model, back='tf', sess=None, params={'eps': 0.3,
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
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        super(FastGradientMethod, self).__init__(model, back, sess, params)

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

    def generate(self, x, params={}):
        """
        Generate symbolic graph for adversarial examples and return.
        """
        if self.back == 'tf':
            from .attacks_tf import fgm
        else:
            from .attacks_th import fgm

        return fgm(x, self.model(x), y=self.y, eps=self.eps, ord=self.ord,
                   clip_min=self.clip_min, clip_max=self.clip_max)

    def generate_np(self, X, params={'Y': None}):
        """
        Generate adversarial samples and return them in a Numpy array.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        # Verify label placeholder was defined previously if using true labels
        if params['Y'] is not None and self.y is None:
            error = "True labels given but label placeholder missing in _init_"
            raise Exception(error)

        import tensorflow as tf

        # Generate this attack's graph if it hasn't been done previously
        if not hasattr(self, "_x"):
            input_shape = list(X.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate(self._x)

        # Run symbolic graph without or with true labels
        if params['Y'] is None:
            feed_dict = {self._x: X}
        else:
            feed_dict = {self._x: X, self.y: params['Y']}
        return self.sess.run(self._x_adv, feed_dict=feed_dict)


class BasicIterativeMethod(Attack):
    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """
    def __init__(self, model, back='tf', sess=None, params={'eps': 0.05,
                                                            'nb_iter': 10,
                                                            'y': None,
                                                            'ord': 'np.inf',
                                                            'clip_min': None,
                                                            'clip_max': None}):
        """
        Create a BasicIterativeMethod instance.

        Attack-specific parameters:
        :param eps: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (required) A placeholder for the model labels.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        super(BasicIterativeMethod, self).__init__(model, back, sess, params)

        # Check that all required attack specific parameters are defined
        req = ('eps', 'nb_iter', 'y')
        if not all(k in params for k in req):
            raise Exception("Attack requires label placeholder.")

        # Save attack-specific parameters
        self.eps = params['eps']
        self.nb_iter = params['nb_iter']
        self.y = params['y']
        self.ord = params['ord'] if 'ord' in params else np.inf
        self.clip_min = params['clip_min'] if 'clip_min' in params else None
        self.clip_max = params['clip_max'] if 'clip_max' in params else None

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise Exception("Norm order must be either np.inf, 1, or 2.")
        if back == 'th':
            error_string = "BasicIterativeMethod is not implemented in Theano"
            raise NotImplementedError(error_string)

    def generate(self, x, params={}):
        import tensorflow as tf

        # Initialize loop variables
        adv_x = x
        y = None
        model_preds = self.model(adv_x)

        for i in range(self.nb_iter):
            FGSM = FastGradientMethod(self.model, back=self.back,
                                      sess=self.sess,
                                      params={'eps': self.eps,
                                              'clip_min': self.clip_min,
                                              'clip_max': self.clip_max,
                                              'y': y})
            adv_x = FGSM.generate(adv_x)
            # After first iteration, we fix the labels to the first model
            # predictions when computing the model loss wrt labels
            if i == 0:
                preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
                y = tf.to_float(tf.equal(model_preds, preds_max))
        return adv_x


class SaliencyMapMethod(Attack):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """
    def __init__(self, model, back='tf', sess=None, params={'theta': 1.,
                                                            'gamma': np.inf,
                                                            'nb_classes': 10,
                                                            'clip_min': 0.,
                                                            'clip_max': 1.,
                                                            'targets': None}):
        """
        Create a SaliencyMapMethod instance.

        Attack-specific parameters:
        :param theta: (required float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (required float) Maximum percentage of perturbed features
        :param nb_classes: (required int) Number of model output classes
        :param clip_min: (required float) Minimum component value for clipping
        :param clip_max: (required float) Maximum component value for clipping
        :param targets: (optional) Target placeholder if the attack is targeted
        """
        super(SaliencyMapMethod, self).__init__(model, back, sess, params)

        if self.back == 'th':
            error = "Theano version of SaliencyMapMethod not implemented."
            raise NotImplementedError(error)

        # Check that all required attack specific parameters are defined
        req = ('theta', 'gamma', 'nb_classes', 'clip_min', 'clip_max')
        if not all(key in params for key in req):
            raise Exception("JSMA must be instantiated w/ params: " + str(req))

        self.theta = params['theta']
        self.gamma = params['gamma']
        self.nb_classes = params['nb_classes']
        self.clip_min = params['clip_min']
        self.clip_max = params['clip_max']
        self.targets = params['targets'] if 'targets' in params else None

    def generate(self, x, params={}):
        """
        Attack-specific parameters:
        """
        import tensorflow as tf
        from .attacks_tf import jacobian_graph, jsma_batch

        # Define Jacobian graph wrt to this input placeholder
        preds = self.model(x)
        grads = jacobian_graph(preds, x, self.nb_classes)

        # Define appropriate graph (targeted / random target labels)
        if self.targets is not None:
            def jsma_wrap(X, targets):
                return jsma_batch(self.sess, x, preds, grads, X, self.theta,
                                  self.gamma, self.clip_min, self.clip_max,
                                  self.nb_classes, targets=targets)

            # Attack is targeted, target placeholder will need to be fed
            wrap = tf.py_func(jsma_wrap, [x, self.targets], tf.float32)
        else:
            def jsma_wrap(X):
                return jsma_batch(self.sess, x, preds, grads, X, self.theta,
                                  self.gamma, self.clip_min, self.clip_max,
                                  self.nb_classes, targets=None)

            # Attack is untargeted, target values will be chosen at random
            wrap = tf.py_func(jsma_wrap, [x], tf.float32)

        return wrap

    def generate_np(self, X, params={'targets': None}):
        """
        Attack-specific parameters:
        :param batch_size: (optional) Batch size when running the graph
        :param targets: (optional) Target values if the attack is targeted
        """
        if 'targets' in params and params['targets'] is not None:
            if self.targets is None:
                raise Exception("This attack was instantiated untargeted.")
            else:
                if len(params['targets'].shape) > 1:
                    nb_targets = len(params['targets'])
                else:
                    nb_targets = 1
                if nb_targets != len(X):
                    raise Exception("Specify exactly one target per input.")
        else:
            params['targets'] = None

        import tensorflow as tf
        # Generate this attack's graph if it hasn't been done previously
        if not hasattr(self, "_x"):
            input_shape = list(X.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate(self._x)

        # Run symbolic graph without or with true labels
        if params['targets'] is None:
            feed_dict = {self._x: X}
        else:
            feed_dict = {self._x: X, self.targets: params['targets']}
        return self.sess.run(self._x_adv, feed_dict=feed_dict)


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
