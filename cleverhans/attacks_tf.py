from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings
import logging

from . import utils_tf
from . import utils

_logger = utils.create_logger("cleverhans.attacks.tf")


def fgsm(x, predictions, eps=0.3, clip_min=None, clip_max=None):
    return fgm(x, predictions, y=None, eps=eps, ord=np.inf, clip_min=clip_min,
               clip_max=clip_max)


def fgm(x, preds, y=None, eps=0.3, ord=np.inf,
        clip_min=None, clip_max=None,
        targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param preds: the model's output tensor (the attack expects the
                  probabilities, i.e., the output of the softmax)
    :param y: (optional) A placeholder for the model labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = utils_tf.model_loss(y, preds, mean=False)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(xrange(1, len(x.get_shape())))
        normalized_grad = grad / tf.reduce_sum(tf.abs(grad),
                                               reduction_indices=red_ind,
                                               keep_dims=True)
    elif ord == 2:
        red_ind = list(xrange(1, len(x.get_shape())))
        square = tf.reduce_sum(tf.square(grad),
                               reduction_indices=red_ind,
                               keep_dims=True)
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def vatm(model, x, logits, eps, num_iterations=1, xi=1e-6,
         clip_min=None, clip_max=None, scope=None):
    """
    Tensorflow implementation of the perturbation method used for virtual
    adversarial training: https://arxiv.org/abs/1507.00677
    :param model: the model which returns the network unnormalized logits
    :param x: the input placeholder
    :param logits: the model's unnormalized output tensor (the input to
                   the softmax layer)
    :param eps: the epsilon (input variation parameter)
    :param num_iterations: the number of iterations
    :param xi: the finite difference parameter
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param seed: the seed for random generator
    :return: a tensor for the adversarial example
    """
    with tf.name_scope(scope, "virtual_adversarial_perturbation"):
        d = tf.random_normal(tf.shape(x))
        for i in range(num_iterations):
            d = xi * utils_tf.l2_batch_normalize(d)
            logits_d = model.get_logits(x + d)
            kl = utils_tf.kl_with_logits(logits, logits_d)
            Hd = tf.gradients(kl, d)[0]
            d = tf.stop_gradient(Hd)
        d = eps * utils_tf.l2_batch_normalize(d)
        adv_x = x + d
        if (clip_min is not None) and (clip_max is not None):
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        return adv_x



def jsma(model, x, y, epochs=1.0, eps=1.0, clip_min=0.0, clip_max=1.0,
         pair=True, min_proba=0.0):
    """Tensorflow implementation of Jacobian-base saliency map approach.

    For details, see https://arxiv.org/abs/1511.07528.  This is the interface
    for JSMA algorithms, detailed implementations are in _jsma_impl and
    _jsma2_impl.

    :param model: A function that returns a tensor output of the model.
    :param x: The input placeholder, a 4d tensor of the shape (batch_size, rows,
              cols, channels).
    :param y: The desired output, either a 0d integer denoting the target class
              or a 2d tensor of the shape (batch_size, target class).
    :param epochs: Number of epochs to run.  If epochs is float, then the max
        iteration is determined with floor(# of pixels / 2 * epochs).
    :param eps: The perturbation applied to image in each epoch.
    :param clip_min: The min value for each pixel.
    :param clip_max: The max value for each pixel.
    :param pair: Perturb two pixels at a time if true, otherwise one pixel at a
        time.  In this paper however, the author recommends to perturb two
        pixels at a time.
    :param min_proba: The minimum confidence that the model makes a desired
        (wrong) prediction.  By default it is 0, which means that as long as the
        adversarial samples can fool the model, we do not care about how strong
        this adversarial samples are.

    :returns: A tensor, the adversarial samples.
    """
    xshape = tf.shape(x)
    n = xshape[0]
    target = tf.cond(tf.equal(0, tf.rank(y)),
                     lambda: tf.zeros([n], dtype=tf.int32)+y,
                     lambda: y)

    if isinstance(epochs, float):
        tmp = tf.to_float(tf.size(x[0])) * epochs
        epochs = tf.to_int32(tf.floor(tmp))

    if pair:
        _jsma_fn = _jsma2_impl
    else:
        _jsma_fn = _jsma_impl

    def _fn(i):
        # `xi` is of the shape (1, ....), the first dimension is the number of
        # samples, 1 in this case.  `yi` is just a scalar, denoting the target
        # class index.
        xi = tf.gather(x, [i])
        yi = tf.gather(target, i)

        # `xadv` is of the shape (1, ...), same as xi.
        xadv = _jsma_fn(model, xi, yi, epochs=epochs, eps=eps,
                        clip_min=clip_min, clip_max=clip_max,
                        min_proba=min_proba)
        return xadv[0]

    return tf.map_fn(_fn, tf.range(n), dtype=tf.float32,
                     back_prop=False, name='jsma_batch')


def _jsma_impl(model, xi, yi, epochs, eps=1.0, clip_min=0.0,
               clip_max=1.0, min_proba=0.0):

    def _cond(x_adv, epoch, pixel_mask):
        ybar = tf.reshape(model(x_adv), [-1])
        proba = ybar[yi]
        label = tf.to_int32(tf.argmax(ybar, axis=0))
        return tf.reduce_all([tf.less(epoch, epochs),
                              tf.reduce_any(pixel_mask),
                              tf.logical_or(tf.not_equal(yi, label),
                                            tf.less(proba, min_proba))],
                             name='_jsma_step_cond')

    def _body(x_adv, epoch, pixel_mask):
        ybar = model(x_adv)

        y_target = tf.slice(ybar, [0, yi], [-1, 1])
        dy_dx, = tf.gradients(ybar, x_adv)

        dt_dx, = tf.gradients(y_target, x_adv)
        do_dx = tf.subtract(dy_dx, dt_dx)
        score = tf.multiply(dt_dx, tf.abs(do_dx))

        cond = tf.logical_and(dt_dx>=0, do_dx<=0)
        domain = tf.logical_and(pixel_mask, cond)
        not_empty = tf.reduce_any(domain)

        # ensure that domain is not empty
        domain, score = tf.cond(not_empty,
                                lambda: (domain, score),
                                lambda: (pixel_mask, dt_dx-do_dx))

        ind = tf.where(domain)
        score = tf.gather_nd(score, ind)

        p = tf.argmax(score, axis=0)
        p = tf.gather(ind, p)
        p = tf.expand_dims(p, axis=0)
        p = tf.to_int32(p)
        dx = tf.scatter_nd(p, [eps], tf.shape(x_adv), name='dx')

        x_adv = tf.stop_gradient(x_adv+dx)
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        epoch += 1
        pixel_mask = tf.cond(tf.greater(eps, 0),
                             lambda: tf.less(x_adv, clip_max),
                             lambda: tf.greater(x_adv, clip_min))

        return x_adv, epoch, pixel_mask

    epoch = tf.Variable(0, tf.int32)
    x_adv = tf.identity(xi)
    pixel_mask = tf.cond(tf.greater(eps, 0),
                         lambda: tf.less(xi, clip_max),
                         lambda: tf.greater(xi, clip_min))

    x_adv, _, _ = tf.while_loop(_cond, _body,
                                (x_adv, epoch, pixel_mask),
                                back_prop=False, name='jsma_step')

    return x_adv


def _jsma2_impl(model, xi, yi, epochs, eps=1.0, clip_min=0.0,
                clip_max=1.0, min_proba=0.0):

    def _cond(x_adv, epoch, pixel_mask):
        ybar = tf.reshape(model(x_adv), [-1])
        proba = ybar[yi]
        label = tf.to_int32(tf.argmax(ybar, axis=0))
        return tf.reduce_all([tf.less(epoch, epochs),
                              tf.reduce_any(pixel_mask),
                              tf.logical_or(tf.not_equal(yi, label),
                                            tf.less(proba, min_proba))],
                             name='_jsma2_step_cond')

    def _body(x_adv, epoch, pixel_mask):
        ybar = model(x_adv)

        y_target = tf.slice(ybar, [0, yi], [-1, 1])
        dy_dx, = tf.gradients(ybar, x_adv)

        dt_dx, = tf.gradients(y_target, x_adv)
        do_dx = dy_dx - dt_dx

        ind = tf.where(pixel_mask)
        n = tf.shape(ind)
        n = n[0]

        ind2 = tf.range(n)
        batch_size = tf.constant(100)

        def _maxpair_batch_cond(i0, j0, v0, start):
            return tf.less(start, n)

        def _maxpair_batch_body(i0, j0, v0, start):
            count = tf.reduce_min([batch_size, n-start])
            ind3 = tf.slice(ind2, [start], [count])

            # Selection C(n, 2), e.g., if n=4, a=[0 0 1 0 1 2], b=[1 2 2 3 3 3],
            # the corresponding element in each array makes a pair, i.e., the
            # pair index are store separately.  A special case is when there is
            # only one pixel left.
            a, b = tf.meshgrid(ind3, ind3)
            c = tf.cond(tf.greater(count, 1),
                        lambda: tf.less(a, b),
                        lambda: tf.less_equal(a, b))
            c = tf.where(c)
            a, b = tf.gather_nd(a, c), tf.gather_nd(b, c)

            # ii, jj contains indices to pixels
            ii, jj = tf.gather(ind, a), tf.gather(ind, b)

            ti, oi = tf.gather_nd(dt_dx, ii), tf.gather_nd(do_dx, ii)
            tj, oj = tf.gather_nd(dt_dx, jj), tf.gather_nd(do_dx, jj)

            # the gradient of each pair is the sum of individuals
            t, o = ti+tj, oi+oj

            # increase target probability while decrease others
            c = tf.logical_and(t>=0, o<=0)
            not_empty = tf.reduce_any(c)

            # ensure that c is not empty
            c = tf.cond(not_empty,
                        lambda: c,
                        lambda: tf.ones_like(c, dtype=bool))
            c = tf.where(c)

            t, o = tf.gather_nd(t, c), tf.gather_nd(o, c)
            ii, jj = tf.gather_nd(ii, c), tf.gather_nd(jj, c)

            # saliency score
            score = tf.cond(not_empty,
                            lambda: tf.multiply(t, tf.abs(o)),
                            lambda: t-o)

            # find the max pair in current batch
            p = tf.argmax(score, axis=0)
            v = tf.reduce_max(score, axis=0)
            i, j = tf.gather(ii, p), tf.gather(jj, p)
            i, j = tf.to_int32(i), tf.to_int32(j)

            i1, j1, v1 = tf.cond(tf.greater(v, v0),
                                 lambda: (i, j, v),
                                 lambda: (i0, j0, v0))
            return i1, j1, v1, start+batch_size

        i = tf.to_int32(tf.gather(ind, 0))
        j = tf.to_int32(tf.gather(ind, 1))
        v = tf.Variable(-1.)
        start = tf.Variable(0)

        # Find max saliency pair in batch.  Naive iteration through the pair
        # takes O(n^2).  Vectorized implementation may speedup the running time
        # significantly, at the expense of O(n^2) space.  So Instead we find the
        # max pair with batch max, during each batch we use vectorized
        # implementation.
        i, j, _, _ = tf.while_loop(_maxpair_batch_cond,
                                   _maxpair_batch_body,
                                   (i, j, v, start), back_prop=False)

        dx = tf.scatter_nd([i], [eps], tf.shape(x_adv)) +\
             tf.scatter_nd([j], [eps], tf.shape(x_adv))

        x_adv = tf.stop_gradient(x_adv+dx)
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        epoch += 1
        pixel_mask = tf.cond(tf.greater(eps, 0),
                             lambda: tf.less(x_adv, clip_max),
                             lambda: tf.greater(x_adv, clip_min))

        return x_adv, epoch, pixel_mask

    epoch = tf.Variable(0, tf.int32)
    x_adv = tf.identity(xi)
    pixel_mask = tf.cond(tf.greater(eps, 0),
                         lambda: tf.less(xi, clip_max),
                         lambda: tf.greater(xi, clip_min))
    x_adv, _, _ = tf.while_loop(_cond, _body, (xi, epoch, pixel_mask),
                                back_prop=False, name='jsma2_step')
    return x_adv


def jacobian_augmentation(sess, x, X_sub_prev, Y_sub, grads, lmbda,
                          keras_phase=None, feed=None):
    """
    Augment an adversary's substitute training set using the Jacobian
    of a substitute model to generate new synthetic inputs.
    See https://arxiv.org/abs/1602.02697 for more details.
    See cleverhans_tutorials/mnist_blackbox.py for example use case
    :param sess: TF session in which the substitute model is defined
    :param x: input TF placeholder for the substitute model
    :param X_sub_prev: substitute training data available to the adversary
                       at the previous iteration
    :param Y_sub: substitute training labels available to the adversary
                  at the previous iteration
    :param grads: Jacobian symbolic graph for the substitute
                  (should be generated using attacks_tf.jacobian_graph)
    :param keras_phase: (deprecated) if not None, holds keras learning_phase
    :return: augmented substitute data (will need to be labeled by oracle)
    """
    assert len(x.get_shape()) == len(np.shape(X_sub_prev))
    assert len(grads) >= np.max(Y_sub) + 1
    assert len(X_sub_prev) == len(Y_sub)

    if keras_phase is not None:
        warnings.warn("keras_phase argument is deprecated and will be removed"
                      " on 2017-09-28. Instead, use K.set_learning_phase(0) at"
                      " the start of your script and serve with tensorflow.")

    # Prepare input_shape (outside loop) for feeding dictionary below
    input_shape = list(x.get_shape())
    input_shape[0] = 1

    # Create new numpy array for adversary training data
    # with twice as many components on the first dimension.
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, input in enumerate(X_sub_prev):
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Prepare feeding dictionary
        feed_dict = {x: np.reshape(input, input_shape)}
        if feed is not None:
            feed_dict.update(feed)

        # Compute sign matrix
        grad_val = sess.run([tf.sign(grad)], feed_dict=feed_dict)[0]

        # Create new synthetic point in adversary substitute training set
        X_sub[2 * ind] = X_sub[ind] + lmbda * grad_val

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub


class CarliniWagnerL2(object):

    def __init__(self, sess, model, batch_size, confidence,
                 targeted, learning_rate,
                 binary_search_steps, max_iterations,
                 abort_early, initial_const,
                 clip_min, clip_max, num_labels, shape):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.

        :param sess: a TF session.
        :param model: a cleverhans.model.Model object.
        :param batch_size: Number of attacks to run simultaneously.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param targeted: boolean controlling the behavior of the adversarial
                         examples produced. If set to False, they will be
                         misclassified in any wrong class. If set to True,
                         they will be misclassified in a chosen target class.
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
        :param clip_min: (optional float) Minimum input component value.
        :param clip_max: (optional float) Maximum input component value.
        :param num_labels: the number of classes in the model's output.
        :param shape: the shape of the model's input tensor.
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = model

        self.repeat = binary_search_steps >= 10

        self.shape = shape = tuple([batch_size] + list(shape))

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32,
                                name='timg')
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)),
                                dtype=tf.float32, name='tlab')
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32,
                                 name='const')

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape,
                                          name='assign_timg')
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels),
                                          name='assign_tlab')
        self.assign_const = tf.placeholder(tf.float32, [batch_size],
                                           name='assign_const')

        # the resulting instance, tanh'd to keep bounded from clip_min
        # to clip_max
        self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
        self.newimg = self.newimg * (clip_max - clip_min) + clip_min

        # prediction BEFORE-SOFTMAX of the model
        self.output = model.get_logits(self.newimg)

        # distance to the input data
        self.other = (tf.tanh(self.timg) + 1) / \
            2 * (clip_max - clip_min) + clip_min
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.other),
                                    list(range(1, len(shape))))

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max(
            (1 - self.tlab) * self.output - self.tlab * 10000,
            1)

        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * loss1)
        self.loss = self.loss1 + self.loss2

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given instance for the given targets.

        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
        for i in range(0, len(imgs), self.batch_size):
            _logger.debug(("Running CWL2 attack on instance " +
                           "{} of {}").format(i, len(imgs)))
            r.extend(self.attack_batch(imgs[i:i + self.batch_size],
                                       targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of instance and labels.
        """
        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        oimgs = np.clip(imgs, self.clip_min, self.clip_max)

        # re-scale instances to be within range [0, 1]
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        # now convert to [-1, 1]
        imgs = (imgs * 2) - 1
        # convert to tanh-space
        imgs = np.arctanh(imgs * .999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(oimgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            _logger.debug("  Binary search step {} of {}".
                          format(outer_step, self.BINARY_SEARCH_STEPS))

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg = self.sess.run([self.train,
                                                         self.loss,
                                                         self.l2dist,
                                                         self.output,
                                                         self.newimg])

                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                                   "l2={:.3g} f={:.3g}")
                                  .format(iteration, self.MAX_ITERATIONS,
                                          l, np.mean(l2s), np.mean(scores)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                   iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                   bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
            _logger.debug("  Successfully generated adversarial examples " +
                          "on {} of {} instances.".
                          format(sum(upper_bound < 1e9), batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
