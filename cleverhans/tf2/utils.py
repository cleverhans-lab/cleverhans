import numpy as np
import tensorflow as tf


def clip_eta(eta, norm, eps):
    """
    Helper function to clip the perturbation to epsilon norm ball.
    :param eta: A tensor with the current perturbation.
    :param norm: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, bound of the perturbation.
    """

    # Clipping perturbation eta to self.norm norm ball
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        eta = tf.clip_by_value(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("")
            # This is not the correct way to project on the L1 norm ball:
            # norm = tf.maximum(avoid_zero_div, reduce_sum(tf.abs(eta), reduc_ind, keepdims=True))
        elif norm == 2:
            # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
            norm = tf.sqrt(
                tf.maximum(
                    avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)
                )
            )
        # We must *clip* to within the norm ball, not *normalize* onto the surface of the ball
        factor = tf.minimum(1.0, tf.math.divide(eps, norm))
        eta = eta * factor
    return eta


def random_exponential(shape, rate=1.0, dtype=tf.float32, seed=None):
    """
    Helper function to sample from the exponential distribution, which is not
    included in core TensorFlow.

    shape: shape of the sampled tensor.
    :rate: (optional) rate parameter of the exponential distribution, defaults to 1.0.
    :dtype: (optional) data type of the sempled tensor, defaults to tf.float32.
    :seed: (optional) custom seed to be used for sampling.
    """
    return tf.random.gamma(shape, alpha=1, beta=1.0 / rate, dtype=dtype, seed=seed)


def random_laplace(shape, loc=0.0, scale=1.0, dtype=tf.float32, seed=None):
    """
    Helper function to sample from the Laplace distribution, which is not
    included in core TensorFlow.

    :shape: shape of the sampled tensor.
    :loc: (optional) mean of the laplace distribution, defaults to 0.0.
    :scale: (optional) scale parameter of the laplace diustribution, defaults to 1.0.
    :dtype: (optional) data type of the sempled tensor, defaults to tf.float32.
    :seed: (optional) custom seed to be used for sampling.
    """
    z1 = random_exponential(shape, 1.0 / scale, dtype=dtype, seed=seed)
    z2 = random_exponential(shape, 1.0 / scale, dtype=dtype, seed=seed)
    return z1 - z2 + loc


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
    """
    Helper function to generate uniformly random vectors from a norm ball of
    radius epsilon.
    :param shape: Output shape of the random sample. The shape is expected to be
                  of the form `(n, d1, d2, ..., dn)` where `n` is the number of
                  i.i.d. samples that will be drawn from a norm ball of dimension
                  `d1*d1*...*dn`.
    :param ord: Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, radius of the norm ball.
    :param dtype: (optional) type of the tensor.
    :param seed: (optional) integer.
    """
    if ord not in [np.inf, 1, 2]:
        raise ValueError("ord must be np.inf, 1, or 2.")

    if ord == np.inf:
        r = tf.random.uniform(shape, -eps, eps, dtype=dtype, seed=seed)
    else:

        # For ord=1 and ord=2, we use the generic technique from
        # (Calafiore et al. 1998) to sample uniformly from a norm ball.
        # Paper link (Calafiore et al. 1998):
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=758215&tag=1
        # We first sample from the surface of the norm ball, and then scale by
        # a factor `w^(1/d)` where `w~U[0,1]` is a standard uniform random variable
        # and `d` is the dimension of the ball. In high dimensions, this is roughly
        # equivalent to sampling from the surface of the ball.

        dim = tf.reduce_prod(shape[1:])

        if ord == 1:
            x = random_laplace(
                (shape[0], dim), loc=1.0, scale=1.0, dtype=dtype, seed=seed
            )
            norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
        elif ord == 2:
            x = tf.random.normal((shape[0], dim), dtype=dtype, seed=seed)
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
        else:
            raise ValueError("ord must be np.inf, 1, or 2.")

        w = tf.pow(
            tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
            1.0 / tf.cast(dim, dtype),
        )
        r = eps * tf.reshape(w * x / norm, shape)

    return r


def get_or_guess_labels(model_fn, x, y=None, targeted=False):
    """
    Helper function to get the label to use in generating an
    adversarial example for x.
    If 'y' is not None, then use these labels.
    If 'targeted' is True, then assume it's a targeted attack
    and y must be set.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    """
    if targeted is True and y is None:
        raise ValueError("Must provide y for a targeted attack!")

    preds = model_fn(x)
    nb_classes = preds.shape[-1]

    # labels set by the user
    if y is not None:
        # inefficient when y is a tensor, but this function only get called once
        y = np.asarray(y)

        if len(y.shape) == 1:
            # the user provided categorical encoding
            y = tf.one_hot(y, nb_classes)

        y = tf.cast(y, x.dtype)
        return y, nb_classes

    # must be an untargeted attack
    labels = tf.cast(
        tf.equal(tf.reduce_max(preds, axis=1, keepdims=True), preds), x.dtype
    )

    return labels, nb_classes


def set_with_mask(x, x_other, mask):
    """Helper function which returns a tensor similar to x with all the values
    of x replaced by x_other where the mask evaluates to true.
    """
    mask = tf.cast(mask, x.dtype)
    ones = tf.ones_like(mask, dtype=x.dtype)
    return x_other * mask + x * (ones - mask)


# Due to performance reasons, this function is wrapped inside of tf.function decorator.
# Not using the decorator here, or letting the user wrap the attack in tf.function is way
# slower on Tensorflow 2.0.0-alpha0.
@tf.function
def compute_gradient(model_fn, loss_fn, x, y, targeted):
    """
    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """

    with tf.GradientTape() as g:
        g.watch(x)
        # Compute loss
        loss = loss_fn(labels=y, logits=model_fn(x))
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, x)
    return grad


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param norm: int specifying order of norm
    :returns:
      tf tensor containing optimal perturbation
    """

    # Convert the iterator returned by `range` into a list.
    axis = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results. It applies only because
        # `optimal_perturbation` is the output of a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the perturbation has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(
            tf.equal(abs_grad, max_abs_grad), dtype=tf.float32
        )
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div, tf.reduce_sum(tf.square(grad), axis, keepdims=True)
        )
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation
