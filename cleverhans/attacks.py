from abc import ABCMeta
import numpy as np
from six.moves import xrange
import warnings
import collections

import cleverhans.utils as utils
from cleverhans.model import Model, CallableModelWrapper

_logger = utils.create_logger("cleverhans.attacks")


class Attack(object):

    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        :param model: An instance of the cleverhans.model.Model class.
        :param back: The backend to use. Currently 'tf' is the only option.
        :param sess: The tf session to run graphs in
        """
        if not(back == 'tf'):
            raise ValueError("Backend argument must either be 'tf'.")

        if back == 'tf':
            import tensorflow as tf
            self.tf_dtype = tf.as_dtype(dtypestr)
            if sess is None:
                sess = tf.get_default_session()

        self.np_dtype = np.dtype(dtypestr)

        import cleverhans.attacks_tf as attacks_tf
        attacks_tf.np_dtype = self.np_dtype
        attacks_tf.tf_dtype = self.tf_dtype

        if not isinstance(model, Model):
            raise ValueError("The model argument should be an instance of"
                             " the cleverhans.model.Model class.")

        # Prepare attributes
        self.model = model
        self.back = back
        self.sess = sess

        # We are going to keep track of old graphs and cache them.
        self.graphs = {}

        # When calling generate_np, arguments in the following set should be
        # fed into the graph, as they are not structural items that require
        # generating a new graph.
        # This dict should map names of arguments to the types they should
        # have.
        # (Usually, the target class will be a feedable keyword argument.)
        self.feedable_kwargs = {}

        # When calling generate_np, arguments in the following set should NOT
        # be fed into the graph, as they ARE structural items that require
        # generating a new graph.
        # This list should contain the names of the structural arguments.
        self.structural_kwargs = []

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

        error = "Sub-classes must implement generate."
        raise NotImplementedError(error)

    def construct_graph(self, fixed, feedable, x_val, hash_key):
        """
        Construct the graph required to run the attack through generate_np.

        :param fixed: Structural elements that require defining a new graph.
        :param feedable: Arguments that can be fed to the same graph when
                         they take different values.
        :param x_val: symbolic adversarial example
        :param hash_key: the key used to store this graph in our cache
        """
        # try our very best to create a TF placeholder for each of the
        # feedable keyword arguments, and check the types are one of
        # the allowed types
        import tensorflow as tf

        class_name = str(self.__class__).split(".")[-1][:-2]
        _logger.info("Constructing new graph for attack " + class_name)

        # remove the None arguments, they are just left blank
        for k in list(feedable.keys()):
            if feedable[k] is None:
                del feedable[k]

        # process all of the rest and create placeholders for them
        new_kwargs = dict(x for x in fixed.items())
        for name, value in feedable.items():
            given_type = self.feedable_kwargs[name]
            if isinstance(value, np.ndarray):
                new_shape = [None] + list(value.shape[1:])
                new_kwargs[name] = tf.placeholder(given_type, new_shape)
            elif isinstance(value, utils.known_number_types):
                new_kwargs[name] = tf.placeholder(given_type, shape=[])
            else:
                raise ValueError("Could not identify type of argument " +
                                 name + ": " + str(value))

        # x is a special placeholder we always want to have
        x_shape = [None] + list(x_val.shape)[1:]
        x = tf.placeholder(self.tf_dtype, shape=x_shape)

        # now we generate the graph that we want
        x_adv = self.generate(x, **new_kwargs)

        self.graphs[hash_key] = (x, new_kwargs, x_adv)

        if len(self.graphs) >= 10:
            warnings.warn("Calling generate_np() with multiple different "
                          "structural paramaters is inefficient and should"
                          " be avoided. Calling generate() is preferred.")

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as a NumPy array.
        Sub-classes *should not* implement this method unless they must
        perform special handling of arguments.

        :param x_val: A NumPy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A NumPy array holding the adversarial examples.
        """
        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")

        fixed, feedable, hash_key = self.construct_variables(kwargs)

        if hash_key not in self.graphs:
            self.construct_graph(fixed, feedable, x_val, hash_key)

        x, new_kwargs, x_adv = self.graphs[hash_key]

        feed_dict = {x: x_val}

        for name in feedable:
            feed_dict[new_kwargs[name]] = feedable[name]

        return self.sess.run(x_adv, feed_dict)

    def construct_variables(self, kwargs):
        """
        Construct the inputs to the attack graph to be used by generate_np.

        :param kwargs: Keyword arguments to generate_np.
        :return: Structural and feedable arguments as well as a unique key
                 for the graph given these inputs.
        """
        # the set of arguments that are structural properties of the attack
        # if these arguments are different, we must construct a new graph
        fixed = dict((k, v) for k, v in kwargs.items()
                     if k in self.structural_kwargs)

        # the set of arguments that are passed as placeholders to the graph
        # on each call, and can change without constructing a new graph
        feedable = dict((k, v) for k, v in kwargs.items()
                        if k in self.feedable_kwargs)

        if len(fixed) + len(feedable) < len(kwargs):
            warnings.warn("Supplied extra keyword arguments that are not "
                          "used in the graph computation. They have been "
                          "ignored.")

        if not all(isinstance(value, collections.Hashable)
                   for value in fixed.values()):
            # we have received a fixed value that isn't hashable
            # this means we can't cache this graph for later use,
            # and it will have to be discarded later
            hash_key = None
        else:
            # create a unique key for this set of fixed paramaters
            hash_key = tuple(sorted(fixed.items()))

        return fixed, feedable, hash_key

    def get_or_guess_labels(self, x, kwargs):
        """
        Get the label to use in generating an adversarial example for x.
        The kwargs are fed directly from the kwargs of the attack.
        If 'y' is in kwargs, then assume it's an untargeted attack and
        use that as the label.
        If 'y_target' is in kwargs, then assume it's a targeted attack and
        use that as the label.
        Otherwise, use the model's prediction as the label and perform an
        untargeted attack.
        """
        import tensorflow as tf

        if 'y' in kwargs and 'y_target' in kwargs:
            raise ValueError("Can not set both 'y' and 'y_target'.")
        elif 'y' in kwargs:
            labels = kwargs['y']
        elif 'y_target' in kwargs:
            labels = kwargs['y_target']
        else:
            preds = self.model.get_probs(x)
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            original_predictions = tf.to_float(tf.equal(preds,
                                                        preds_max))
            labels = tf.stop_gradient(original_predictions)
        if isinstance(labels, np.ndarray):
            nb_classes = labels.shape[1]
        else:
            nb_classes = labels.get_shape().as_list()[1]
        return labels, nb_classes

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

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Create a FastGradientMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(FastGradientMethod, self).__init__(model, back, sess, dtypestr)
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'y': self.np_dtype,
                                'y_target': self.np_dtype,
                                'clip_min': self.np_dtype,
                                'clip_max': self.np_dtype}
        self.structural_kwargs = ['ord']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        from .attacks_tf import fgm

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        return fgm(x, self.model.get_probs(x), y=labels, eps=self.eps,
                   ord=self.ord, clip_min=self.clip_min,
                   clip_max=self.clip_max,
                   targeted=(self.y_target is not None))

    def parse_params(self, eps=0.3, ord=np.inf, y=None, y_target=None,
                     clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True


class BasicIterativeMethod(Attack):

    """
    The Basic Iterative Method (Kurakin et al. 2016). The original paper used
    hard labels for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Create a BasicIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(BasicIterativeMethod, self).__init__(model, back, sess, dtypestr)
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'eps_iter': self.np_dtype,
                                'y': self.np_dtype,
                                'y_target': self.np_dtype,
                                'clip_min': self.np_dtype,
                                'clip_max': self.np_dtype}
        self.structural_kwargs = ['ord', 'nb_iter']

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
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        eta = 0

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_probs(x)
        preds_max = tf.reduce_max(model_preds, 1, keep_dims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {'eps': self.eps_iter, y_kwarg: y, 'ord': self.ord,
                      'clip_min': self.clip_min, 'clip_max': self.clip_max}

        for i in range(self.nb_iter):
            FGM = FastGradientMethod(self.model, back=self.back,
                                     sess=self.sess)
            # Compute this step's perturbation
            adv_x = FGM.generate(x + eta, **fgm_params)

            # Clipping perturbation according to clip_min and clip_max
            if self.clip_min is not None and self.clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

            # Clipping perturbation eta to self.ord norm ball
            eta = adv_x - x
            from cleverhans.utils_tf import clip_eta
            eta = clip_eta(eta, self.ord, self.eps)

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.05, nb_iter=10, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
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

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True


class MomentumIterativeMethod(Attack):

    """
    The Momentum Iterative Method (Dong et al. 2017). This method won
    the first places in NIPS 2017 Non-targeted Adversarial Attacks and
    Targeted Adversarial Attacks. The original paper used hard labels
    for this attack; no label smoothing.
    Paper link: https://arxiv.org/pdf/1710.06081.pdf
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Create a MomentumIterativeMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(MomentumIterativeMethod, self).__init__(model, back,
                                                      sess, dtypestr)
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'eps_iter': self.np_dtype,
                                'y': self.np_dtype,
                                'y_target': self.np_dtype,
                                'clip_min': self.np_dtype,
                                'clip_max': self.np_dtype}
        self.structural_kwargs = ['ord', 'nb_iter', 'decay_factor']

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
        :param decay_factor: (optional) Decay factor for the momentum term.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        momentum = 0
        adv_x = x

        # Fix labels to the first model predictions for loss computation
        y, nb_classes = self.get_or_guess_labels(x, kwargs)
        y = y / tf.reduce_sum(y, 1, keep_dims=True)
        targeted = (self.y_target is not None)

        from . import utils_tf
        for i in range(self.nb_iter):
            # Compute loss
            preds = self.model.get_probs(adv_x)
            loss = utils_tf.model_loss(y, preds, mean=False)
            if targeted:
                loss = -loss

            # Define gradient of loss wrt input
            grad, = tf.gradients(loss, adv_x)

            # Normalize current gradient and add it to the accumulated gradient
            red_ind = list(xrange(1, len(grad.get_shape())))
            avoid_zero_div = tf.cast(1e-12, grad.dtype)
            grad = grad / tf.maximum(avoid_zero_div,
                                     tf.reduce_mean(tf.abs(grad),
                                                    red_ind,
                                                    keep_dims=True))
            momentum = self.decay_factor * momentum + grad

            if self.ord == np.inf:
                normalized_grad = tf.sign(momentum)
            elif self.ord == 1:
                norm = tf.maximum(avoid_zero_div,
                                  tf.reduce_sum(tf.abs(momentum),
                                                red_ind,
                                                keep_dims=True))
                normalized_grad = momentum / norm
            elif self.ord == 2:
                square = tf.reduce_sum(tf.square(momentum),
                                       red_ind,
                                       keep_dims=True)
                norm = tf.sqrt(tf.maximum(avoid_zero_div, square))
                normalized_grad = momentum / norm
            else:
                raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                          "currently implemented.")

            # Update and clip adversarial example in current iteration
            scaled_grad = self.eps_iter * normalized_grad
            adv_x = adv_x + scaled_grad
            adv_x = x + utils_tf.clip_eta(adv_x - x, self.ord, self.eps)

            if self.clip_min is not None and self.clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

            adv_x = tf.stop_gradient(adv_x)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.06, nb_iter=10, y=None,
                     ord=np.inf, decay_factor=1.0,
                     clip_min=None, clip_max=None,
                     y_target=None, **kwargs):
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
        :param decay_factor: (optional) Decay factor for the momentum term.
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
        self.decay_factor = decay_factor
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True


class SaliencyMapMethod(Attack):

    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Create a SaliencyMapMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(SaliencyMapMethod, self).__init__(model, back, sess, dtypestr)

        import tensorflow as tf
        self.feedable_kwargs = {'y_target': self.tf_dtype}
        self.structural_kwargs = ['theta', 'gamma',
                                  'clip_max', 'clip_min', 'symbolic_impl']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param theta: (optional float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :param y_target: (optional) Target tensor if the attack is targeted
        """
        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        if self.symbolic_impl:
            from .attacks_tf import jsma_symbolic

            # Create random targets if y_target not provided
            if self.y_target is None:
                from random import randint

                def random_targets(gt):
                    result = gt.copy()
                    nb_s = gt.shape[0]
                    nb_classes = gt.shape[1]

                    for i in xrange(nb_s):
                        result[i, :] = np.roll(result[i, :],
                                               randint(1, nb_classes-1))

                    return result

                labels, nb_classes = self.get_or_guess_labels(x, kwargs)
                self.y_target = tf.py_func(random_targets, [labels],
                                           self.tf_dtype)
                self.y_target.set_shape([None, nb_classes])

            x_adv = jsma_symbolic(x, model=self.model, y_target=self.y_target,
                                  theta=self.theta, gamma=self.gamma,
                                  clip_min=self.clip_min,
                                  clip_max=self.clip_max)
        else:
            from .attacks_tf import jacobian_graph, jsma_batch

            # Define Jacobian graph wrt to this input placeholder
            preds = self.model.get_probs(x)
            nb_classes = preds.get_shape().as_list()[-1]
            grads = jacobian_graph(preds, x, nb_classes)

            # Define appropriate graph (targeted / random target labels)
            if self.y_target is not None:
                def jsma_wrap(x_val, y_target):
                    return jsma_batch(self.sess, x, preds, grads, x_val,
                                      self.theta, self.gamma, self.clip_min,
                                      self.clip_max, nb_classes,
                                      y_target=y_target)

                # Attack is targeted, target placeholder will need to be fed
                x_adv = tf.py_func(jsma_wrap, [x, self.y_target],
                                   self.tf_dtype)
            else:
                def jsma_wrap(x_val):
                    return jsma_batch(self.sess, x, preds, grads, x_val,
                                      self.theta, self.gamma, self.clip_min,
                                      self.clip_max, nb_classes,
                                      y_target=None)

                # Attack is untargeted, target values will be chosen at random
                x_adv = tf.py_func(jsma_wrap, [x], self.tf_dtype)

        return x_adv

    def parse_params(self, theta=1., gamma=1., nb_classes=None,
                     clip_min=0., clip_max=1., y_target=None,
                     symbolic_impl=True, **kwargs):
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
        :param y_target: (optional) Target tensor if the attack is targeted
        """

        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y_target = y_target
        self.symbolic_impl = symbolic_impl

        return True


class VirtualAdversarialMethod(Attack):

    """
    This attack was originally proposed by Miyato et al. (2016) and was used
    for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677

    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')

        super(VirtualAdversarialMethod, self).__init__(model, back,
                                                       sess, dtypestr)

        import tensorflow as tf
        self.feedable_kwargs = {'eps': self.tf_dtype, 'xi': self.tf_dtype,
                                'clip_min': self.tf_dtype,
                                'clip_max': self.tf_dtype}
        self.structural_kwargs = ['num_iterations']

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

        return vatm(self.model, x, self.model.get_logits(x), eps=self.eps,
                    num_iterations=self.num_iterations, xi=self.xi,
                    clip_min=self.clip_min, clip_max=self.clip_max)

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


class CarliniWagnerL2(Attack):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644

    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.
    """
    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')

        super(CarliniWagnerL2, self).__init__(model, back, sess, dtypestr)

        import tensorflow as tf
        self.feedable_kwargs = {'y': self.tf_dtype,
                                'y_target': self.tf_dtype}

        self.structural_kwargs = ['batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf import CarliniWagnerL2 as CWL2
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CWL2(self.sess, self.model, self.batch_size,
                      self.confidence, 'y_target' in kwargs,
                      self.learning_rate, self.binary_search_steps,
                      self.max_iterations, self.abort_early,
                      self.initial_const, self.clip_min, self.clip_max,
                      nb_classes, x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)
        wrap = tf.py_func(cw_wrap, [x, labels], self.tf_dtype)

        return wrap

    def parse_params(self, y=None, y_target=None, nb_classes=None,
                     batch_size=1, confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5, max_iterations=1000,
                     abort_early=True, initial_const=1e-2,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max


class ElasticNetMethod(Attack):
    """
    This attack features L1-oriented adversarial examples and includes
    the C&W L2 attack as a special case (when beta is set to 0).
    Adversarial examples attain similar performance to those
    generated by the C&W L2 attack in the white-box case,
    and more importantly, have improved transferability properties
    and complement adversarial training.
    Paper link: https://arxiv.org/abs/1709.04114
    """
    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')

        super(ElasticNetMethod, self).__init__(model, back, sess, dtypestr)

        import tensorflow as tf
        self.feedable_kwargs = {'y': self.tf_dtype,
                                'y_target': self.tf_dtype}

        self.structural_kwargs = ['fista', 'beta', 'decision_rule',
                                  'batch_size', 'confidence',
                                  'targeted', 'learning_rate',
                                  'binary_search_steps', 'max_iterations',
                                  'abort_early', 'initial_const',
                                  'clip_min', 'clip_max']

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param fista: FISTA or ISTA. FISTA has better convergence properties
                      but performs an additional query per iteration
        :param beta: Trades off L2 distortion with L1 distortion: higher
                     produces examples with lower L1 distortion, at the
                     cost of higher L2 (and typically Linf) distortion
        :param decision_rule: EN or L1. Select final adversarial example from
                              all successful examples based on the least
                              elastic-net or L1 distortion criterion.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the perturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early abort when the total
                            loss starts to increase (greatly speeds up attack,
                            but hurts performance, particularly on ImageNet)
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the perturbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        self.parse_params(**kwargs)

        from .attacks_tf import ElasticNetMethod as EAD
        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = EAD(self.sess, self.model, self.fista, self.beta,
                     self.decision_rule, self.batch_size, self.confidence,
                     'y_target' in kwargs, self.learning_rate,
                     self.binary_search_steps, self.max_iterations,
                     self.abort_early, self.initial_const,
                     self.clip_min, self.clip_max,
                     nb_classes, x.get_shape().as_list()[1:])

        def ead_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)
        wrap = tf.py_func(ead_wrap, [x, labels], self.tf_dtype)

        return wrap

    def parse_params(self, y=None, y_target=None,
                     nb_classes=None, fista=True, beta=1e-3,
                     decision_rule='EN', batch_size=1, confidence=0,
                     learning_rate=1e-2,
                     binary_search_steps=9, max_iterations=1000,
                     abort_early=False, initial_const=1e-3,
                     clip_min=0, clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.fista = fista
        self.beta = beta
        self.decision_rule = decision_rule
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max


class DeepFool(Attack):

    """
    DeepFool is an untargeted & iterative attack which is based on an
    iterative linearization of the classifier. The implementation here
    is w.r.t. the L2 norm.
    Paper link: "https://arxiv.org/pdf/1511.04599.pdf"
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Create a DeepFool instance.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')

        super(DeepFool, self).__init__(model, back, sess, dtypestr)

        self.structural_kwargs = ['over_shoot', 'max_iter', 'clip_max',
                                  'clip_min', 'nb_candidate']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param nb_candidate: The number of classes to test against, i.e.,
                             deepfool only consider nb_candidate classes when
                             attacking(thus accelerate speed). The nb_candidate
                             classes are chosen according to the prediction
                             confidence during implementation.
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for deepfool
        :param nb_classes: The number of model output classes
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        """

        import tensorflow as tf
        from .attacks_tf import jacobian_graph, deepfool_batch

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Define graph wrt to this input placeholder
        logits = self.model.get_logits(x)
        self.nb_classes = logits.get_shape().as_list()[-1]
        assert self.nb_candidate <= self.nb_classes,\
            'nb_candidate should not be greater than nb_classes'
        preds = tf.reshape(tf.nn.top_k(logits, k=self.nb_candidate)[0],
                           [-1, self.nb_candidate])
        # grads will be the shape [batch_size, nb_candidate, image_size]
        grads = tf.stack(jacobian_graph(preds, x, self.nb_candidate), axis=1)

        # Define graph
        def deepfool_wrap(x_val):
            return deepfool_batch(self.sess, x, preds, logits, grads, x_val,
                                  self.nb_candidate, self.overshoot,
                                  self.max_iter, self.clip_min, self.clip_max,
                                  self.nb_classes)
        return tf.py_func(deepfool_wrap, [x], self.tf_dtype)

    def parse_params(self, nb_candidate=10, overshoot=0.02, max_iter=50,
                     nb_classes=None, clip_min=0., clip_max=1., **kwargs):
        """
        :param nb_candidate: The number of classes to test against, i.e.,
                             deepfool only consider nb_candidate classes when
                             attacking(thus accelerate speed). The nb_candidate
                             classes are chosen according to the prediction
                             confidence during implementation.
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for deepfool
        :param nb_classes: The number of model output classes
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        """
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max

        return True


class LBFGS(Attack):
    """
    LBFGS is the first adversarial attack for convolutional neural networks,
    and is a target & iterative attack.
    Paper link: "https://arxiv.org/pdf/1312.6199.pdf"
    """
    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(LBFGS, self).__init__(model, back, sess, dtypestr)

        import tensorflow as tf
        self.feedable_kwargs = {'y_target': self.tf_dtype}
        self.structural_kwargs = ['batch_size', 'binary_search_steps',
                                  'max_iterations', 'initial_const',
                                  'clip_min', 'clip_max']

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param x: (required) A tensor with the inputs.
        :param y_target: (required) A tensor with the one-hot target labels.
        :param batch_size: The number of inputs to include in a batch and
                           process simultaneously.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and cross-entropy loss of classification.
        :param max_iterations: The maximum number of iterations.
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the perturbation
                              and cross-entropy loss of the classification.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from .attacks_tf import LBFGS_attack
        self.parse_params(**kwargs)

        _, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = LBFGS_attack(self.sess, x, self.model.get_probs(x),
                              self.y_target, self.binary_search_steps,
                              self.max_iterations, self.initial_const,
                              self.clip_min, self.clip_max, nb_classes,
                              self.batch_size)

        def lbfgs_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)
        wrap = tf.py_func(lbfgs_wrap, [x, self.y_target], self.tf_dtype)

        return wrap

    def parse_params(self, y_target=None, batch_size=1,
                     binary_search_steps=5, max_iterations=1000,
                     initial_const=1e-2, clip_min=0, clip_max=1):

        self.y_target = y_target
        self.batch_size = batch_size
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max


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
    assert back == 'tf'
    # Compute VATM using TensorFlow
    from .attacks_tf import vatm as vatm_tf
    return vatm_tf(model, x, logits, eps, num_iterations=num_iterations,
                   xi=xi, clip_min=clip_min, clip_max=clip_max)


class MadryEtAl(Attack):

    """
    The Projected Gradient Descent Attack (Madry et al. 2017).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Create a MadryEtAl instance.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(MadryEtAl, self).__init__(model, back, sess, dtypestr)
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'eps_iter': self.np_dtype,
                                'y': self.np_dtype,
                                'y_target': self.np_dtype,
                                'clip_min': self.np_dtype,
                                'clip_max': self.np_dtype}
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init']

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
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                     ord=np.inf, clip_min=None, clip_max=None,
                     y_target=None, rand_init=True, **kwargs):
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
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
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
        self.rand_init = rand_init

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import model_loss, clip_eta

        adv_x = x + eta
        preds = self.model.get_probs(adv_x)
        loss = model_loss(y, preds)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.

        :param x: A tensor with the input image.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        if self.rand_init:
            eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps,
                                    dtype=self.tf_dtype)
            eta = clip_eta(eta, self.ord, self.eps)
        else:
            eta = tf.zeros_like(x)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, y)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x


class FastFeatureAdversaries(Attack):
    """
    This is a fast implementation of "Feature Adversaries", an attack
    against a target internal representation of a model.
    "Feature adversaries" were originally introduced in (Sabour et al. 2016),
    where the optimization was done using LBFGS.
    Paper link: https://arxiv.org/abs/1511.05122

    This implementation is similar to "Basic Iterative Method"
    (Kurakin et al. 2016) but applied to the internal representations.
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        Create a FastFeatureAdversaries instance.
        """
        super(FastFeatureAdversaries, self).__init__(model, back,
                                                     sess, dtypestr)
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'eps_iter': self.np_dtype,
                                'clip_min': self.np_dtype,
                                'clip_max': self.np_dtype,
                                'layer': str}
        self.structural_kwargs = ['ord', 'nb_iter']

        assert isinstance(self.model, Model)

    def parse_params(self, layer=None, eps=0.3, eps_iter=0.05, nb_iter=10,
                     ord=np.inf, clip_min=None, clip_max=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param layer: (required str) name of the layer to target.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.layer = layer
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def attack_single_step(self, x, eta, g_feat):
        """
        TensorFlow implementation of the Fast Feature Gradient. This is a
        single step attack similar to Fast Gradient Method that attacks an
        internal representation.

        :param x: the input placeholder
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param g_feat: model's internal tensor for guide
        :return: a tensor for the adversarial example
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        adv_x = x + eta
        a_feat = self.model.get_layer(adv_x, self.layer)

        # feat.shape = (batch, c) or (batch, w, h, c)
        axis = list(range(1, len(a_feat.shape)))

        # Compute loss
        # This is a targeted attack, hence the negative sign
        loss = -tf.reduce_sum(tf.square(a_feat - g_feat), axis)

        # Define gradient of loss wrt input
        grad, = tf.gradients(loss, adv_x)

        # Multiply by constant epsilon
        scaled_signed_grad = self.eps_iter * tf.sign(grad)

        # Add perturbation to original example to obtain adversarial example
        adv_x = adv_x + scaled_signed_grad

        # If clipping is needed,
        # reset all values outside of [clip_min, clip_max]
        if (self.clip_min is not None) and (self.clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        adv_x = tf.stop_gradient(adv_x)

        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)

        return eta

    def generate(self, x, g, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param g: The target's symbolic representation.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        g_feat = self.model.get_layer(g, self.layer)

        # Initialize loop variables
        eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps,
                                dtype=self.tf_dtype)
        eta = clip_eta(eta, self.ord, self.eps)

        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, g_feat)

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x


class SPSA(Attack):
    """
    This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
    (Uesato et al. 2018). SPSA is a gradient-free optimization method, which
    is useful when the model is non-differentiable, or more generally, the
    gradients do not point in useful directions.
    """

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        super(SPSA, self).__init__(model, back, sess, dtypestr)
        assert isinstance(self.model, Model)

    def generate(self, x, y=None, y_target=None, epsilon=None, num_steps=None,
                 is_targeted=False, early_stop_loss_threshold=None,
                 learning_rate=0.01, delta=0.01, batch_size=128, spsa_iters=1,
                 is_debug=False):
        """
        Generate symbolic graph for adversarial examples.

        :param x: The model's symbolic inputs. Must be a batch of size 1.
        :param y: A Tensor or None. The index of the correct label.
        :param y_target: A Tensor or None. The index of the target label in a
                         targeted attack.
        :param epsilon: The size of the maximum perturbation, measured in the
                        L-infinity norm.
        :param num_steps: The number of optimization steps.
        :param is_targeted: Whether to use a targeted or untargeted attack.
        :param early_stop_loss_threshold: A float or None. If specified, the
                                          attack will end as soon as the loss
                                          is below `early_stop_loss_threshold`.
        :param learning_rate: Learning rate of ADAM optimizer.
        :param delta: Perturbation size used for SPSA approximation.
        :param batch_size: Number of inputs to evaluate at a single time. Note
                           that the true batch size (the number of evaluated
                           inputs for each update) is `batch_size * spsa_iters`
        :param spsa_iters: Number of model evaluations before performing an
                           update, where each evaluation is on `batch_size`
                           different inputs.
        :param is_debug: If True, print the adversarial loss after each update.
        """
        from .attacks_tf import SPSAAdam, pgd_attack, margin_logit_loss

        optimizer = SPSAAdam(lr=learning_rate, delta=delta,
                             num_samples=batch_size, num_iters=spsa_iters)

        def loss_fn(x, label):
            logits = self.model.get_logits(x)
            loss_multiplier = 1 if is_targeted else -1
            return loss_multiplier * margin_logit_loss(
                logits, label, num_classes=self.model.num_classes)

        y_attack = y_target if is_targeted else y
        adv_x = pgd_attack(
            loss_fn, x, y_attack, epsilon, num_steps=num_steps,
            optimizer=optimizer,
            early_stop_loss_threshold=early_stop_loss_threshold,
            is_debug=is_debug,
        )
        return adv_x
