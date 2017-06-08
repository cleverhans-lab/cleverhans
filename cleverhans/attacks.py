from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings


class Attack(object):

    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None):
        """
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        :param back: The backend to use. Either 'tf' (default) or 'th'.
        :param sess: The tf session to run graphs in (use None for Theano)
        """
        if not(back == 'tf' or back == 'th'):
            raise ValueError("Backend argument must either be 'tf' or 'th'.")
        if back == 'th' and sess is not None:
            raise Exception("A session should not be provided when using th.")
        if not hasattr(model, '__call__'):
            raise ValueError("model argument must be a function that returns "
                             "the symbolic output when given an input tensor.")
        if back == 'th':
            warnings.warn("CleverHans support for Theano is deprecated and "
                          "will be dropped on 2017-11-08.")

        # Prepare attributes
        self.model = model
        self.back = back
        self.sess = sess
        self.inf_loop = False

    def generate(self, x, **kwargs):
        """
        Generate the attack's symbolic graph for adversarial examples. This
        method should be overriden in any child class that implements an
        attack that is expressable symbolically. Otherwise, it will wrap the
        numerical implementation as a symbolic operator.
        :param x: The model's symbolic inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A symbolic representation of the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        if not self.inf_loop:
            self.inf_loop = True
            assert self.parse_params(**kwargs)
            import tensorflow as tf
            graph = tf.py_func(self.generate_np, [x], tf.float32)
            self.inf_loop = False
            return graph
        else:
            error = "No symbolic or numeric implementation of attack."
            raise NotImplementedError(error)

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as a Numpy array. This
        method should be overriden in any child class that implements an attack
        that is not fully expressed symbolically.
        :param x_val: A Numpy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A Numpy array holding the adversarial examples.
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        if not self.inf_loop:
            self.inf_loop = True
            import tensorflow as tf

            # Generate this attack's graph if not done previously
            if not hasattr(self, "_x") and not hasattr(self, "_x_adv"):
                input_shape = list(x_val.shape)
                input_shape[0] = None
                self._x = tf.placeholder(tf.float32, shape=input_shape)
                self._x_adv = self.generate(self._x, **kwargs)
            self.inf_loop = False
        else:
            error = "No symbolic or numeric implementation of attack."
            raise NotImplementedError(error)

        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")
        return self.sess.run(self._x_adv, feed_dict={self._x: x_val})

    def parse_params(self, params=None):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        :param params: a dictionary of attack-specific parameters
        :return: True when parsing was successful
        """
        return True


class FastGradientMethod(Attack):

    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a FastGradientMethod instance.
        """
        super(FastGradientMethod, self).__init__(model, back, sess)

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
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
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        if self.back == 'tf':
            from .attacks_tf import fgm
        else:
            from .attacks_th import fgm

        return fgm(x, self.model(x), y=self.y, eps=self.eps, ord=self.ord,
                   clip_min=self.clip_min, clip_max=self.clip_max)

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
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
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')
        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")

        import tensorflow as tf

        # Generate this attack's graph if it hasn't been done previously
        if not hasattr(self, "_x"):
            input_shape = list(x_val.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate(self._x, **kwargs)

        # Run symbolic graph without or with true labels
        if 'y_val' not in kwargs or kwargs['y_val'] is None:
            feed_dict = {self._x: x_val}
        else:
            # Verify label placeholder was given in params if using true labels
            if self.y is None:
                error = "True labels given but label placeholder not given."
                raise Exception(error)
            feed_dict = {self._x: x_val, self.y: kwargs['y_val']}
        return self.sess.run(self._x_adv, feed_dict=feed_dict)

    def parse_params(self, eps=0.3, ord=np.inf, y=None, clip_min=None,
                     clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
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
        # Save attack-specific parameters
        self.eps = eps
        self.ord = ord
        self.y = y
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th' and self.ord != np.inf:
            raise NotImplementedError("The only FastGradientMethod norm "
                                      "implemented for Theano is np.inf.")
        return True


class BasicIterativeMethod(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a BasicIterativeMethod instance.
        """
        super(BasicIterativeMethod, self).__init__(model, back, sess)

    def generate(self, x, **kwargs):
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(model_preds, preds_max))
        fgsm_params = {'eps': self.eps_iter, 'y': y, 'ord': self.ord}

        for i in range(self.nb_iter):
            FGSM = FastGradientMethod(self.model, back=self.back,
                                      sess=self.sess)
            # Compute this step's perturbation
            eta = FGSM.generate(x + eta, **fgsm_params) - x

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

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (required) A placeholder for the model labels.
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
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th':
            error_string = "BasicIterativeMethod is not implemented in Theano"
            raise NotImplementedError(error_string)

        return True


class SaliencyMapMethod(Attack):

    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a SaliencyMapMethod instance.
        """
        super(SaliencyMapMethod, self).__init__(model, back, sess)

        if self.back == 'th':
            error = "Theano version of SaliencyMapMethod not implemented."
            raise NotImplementedError(error)

    def generate(self, x, **kwargs):
        """
        Attack-specific parameters:
        """
        import tensorflow as tf
        from .attacks_tf import jacobian_graph, jsma_batch

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Define Jacobian graph wrt to this input placeholder
        preds = self.model(x)
        grads = jacobian_graph(preds, x, self.nb_classes)

        # Define appropriate graph (targeted / random target labels)
        if self.targets is not None:
            def jsma_wrap(x_val, targets):
                return jsma_batch(self.sess, x, preds, grads, x_val,
                                  self.theta, self.gamma, self.clip_min,
                                  self.clip_max, self.nb_classes,
                                  targets=targets)

            # Attack is targeted, target placeholder will need to be fed
            wrap = tf.py_func(jsma_wrap, [x, self.targets], tf.float32)
        else:
            def jsma_wrap(x_val):
                return jsma_batch(self.sess, x, preds, grads, x_val,
                                  self.theta, self.gamma, self.clip_min,
                                  self.clip_max, self.nb_classes,
                                  targets=None)

            # Attack is untargeted, target values will be chosen at random
            wrap = tf.py_func(jsma_wrap, [x], tf.float32)

        return wrap

    def generate_np(self, x_val, **kwargs):
        """
        Attack-specific parameters:
        :param batch_size: (optional) Batch size when running the graph
        :param targets: (optional) Target values if the attack is targeted
        """
        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")

        import tensorflow as tf

        # Generate this attack's graph if it hasn't been done previously
        if not hasattr(self, "_x"):
            input_shape = list(x_val.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate(self._x, **kwargs)

        # Run symbolic graph without or with true labels
        if 'y_val' not in kwargs or kwargs['y_val'] is None:
            feed_dict = {self._x: x_val}
        else:
            if self.targets is None:
                raise Exception("This attack was instantiated untargeted.")
            else:
                if len(kwargs['y_val'].shape) > 1:
                    nb_targets = len(kwargs['y_val'])
                else:
                    nb_targets = 1
                if nb_targets != len(x_val):
                    raise Exception("Specify exactly one target per input.")
            feed_dict = {self._x: x_val, self.targets: kwargs['y_val']}
        return self.sess.run(self._x_adv, feed_dict=feed_dict)

    def parse_params(self, theta=1., gamma=np.inf, nb_classes=10, clip_min=0.,
                     clip_max=1., targets=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param theta: (optional float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param nb_classes: (optional int) Number of model output classes
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :param targets: (optional) Target placeholder if the attack is targeted
        """

        self.theta = theta
        self.gamma = gamma
        self.nb_classes = nb_classes
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targets = targets

        return True


class VirtualAdversarialMethod(Attack):

    """
    This attack was originally proposed by Miyato et al. (2016) and was used
    for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677

    """

    def __init__(self, model, back='tf', sess=None):
        super(VirtualAdversarialMethod, self).__init__(model, back, sess)

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float ) the epsilon (input variation parameter)
        :param num_iterations: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        return vatm(self.model, x, self.model(x), eps=self.eps,
                    num_iterations=self.num_iterations, xi=self.xi,
                    clip_min=self.clip_min, clip_max=self.clip_max)

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :param eps: (optional float )the epsilon (input variation parameter)
        :param num_iterations: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        import tensorflow as tf

        # Generate this attack's graph if it hasn't been done previously
        if not hasattr(self, "_x"):
            input_shape = list(x_val.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate(self._x, **kwargs)

        return self.sess.run(self._x_adv, feed_dict={self._x: x_val})

    def parse_params(self, eps=2.0, num_iterations=1, xi=1e-6, clip_min=None,
                     clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float )the epsilon (input variation parameter)
        :param num_iterations: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        self.eps = eps
        self.num_iterations = num_iterations
        self.xi = xi
        self.clip_min = clip_min
        self.clip_max = clip_max
        return True


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
    warnings.warn("attacks.fgsm is deprecated and will be removed on "
                  "2017-09-27. Instantiate an object from FastGradientMethod.")
    if back == 'tf':
        # Compute FGSM using TensorFlow
        from .attacks_tf import fgm
        return fgm(x, predictions, y=None, eps=eps, ord=np.inf,
                   clip_min=clip_min, clip_max=clip_max)
    elif back == 'th':
        # Compute FGSM using Theano
        from .attacks_th import fgm
        return fgm(x, predictions, eps, clip_min=clip_min, clip_max=clip_max)


def vatm(model, x, logits, eps, back='tf', num_iterations=1, xi=1e-6,
         clip_min=None, clip_max=None):
    """
    A wrapper for the perturbation methods used for virtual adversarial
    training : https://arxiv.org/abs/1507.00677
    It calls the right function, depending on the
    user's backend.
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example

    """
    if back == 'tf':
        # Compute VATM using TensorFlow
        from .attacks_tf import vatm as vatm_tf
        return vatm_tf(model, x, logits, eps, num_iterations=num_iterations,
                       xi=xi, clip_min=clip_min, clip_max=clip_max)
    elif back == 'th':
        # Compute VATM using Theano
        from .attacks_th import vatm as vatm_th
        return vatm_th(model, x, logits, eps, num_iterations=num_iterations,
                       xi=xi, clip_min=clip_min, clip_max=clip_max)


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
    warnings.warn("attacks.jsma is deprecated and will be removed on "
                  "2017-09-27. Instantiate an object from SaliencyMapMethod.")
    if back == 'tf':
        # Compute Jacobian-based saliency map attack using TensorFlow
        from .attacks_tf import jsma
        return jsma(sess, x, predictions, grads, sample, target, theta, gamma,
                    clip_min, clip_max)
    elif back == 'th':
        raise NotImplementedError("Theano jsma not implemented.")
