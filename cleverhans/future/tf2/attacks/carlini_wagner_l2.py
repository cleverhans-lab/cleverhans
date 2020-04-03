"""The CarliniWagnerL2 attack.
"""
import numpy as np
import tensorflow as tf

from cleverhans.future.tf2.utils_tf import get_or_guess_labels


def carlini_wagner_l2(model_fn, x, y=None, targeted=False, clip_min=0., clip_max=1., binary_search_steps=5, max_iterations=1_000, abort_early=True, confidence=0., initial_const=1e-2, learning_rate=5e-3):
    """
    This attack was originally proposed by Carlini and Wagner. It is an
    iterative attack that finds adversarial examples on many defenses that
    are robust to other attacks.
    Paper link: https://arxiv.org/abs/1608.04644
    At a high level, this attack is an iterative attack using Adam and
    a specially-chosen loss function to find adversarial examples with
    lower distortion than other attacks. This comes at the cost of speed,
    as this attack is often much slower than others.

    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the default,
            will try to make the label incorrect. Targeted will instead try to move in the direction
            of being more like y.
    :param clip_min: (optional) float. Minimum float values for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
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
    :param confidence: Confidence of adversarial examples: higher produces
                       examples with larger l2 distortion, but more
                       strongly classified as adversarial.
    :param initial_const: The initial tradeoff-constant to use to tune the
                      relative importance of size of the pururbation
                      and confidence of classification.
                      If binary_search_steps is large, the initial
                      constant is not important. A smaller value of
                      this constant gives lower distortion results.
    :param learning_rate: The learning rate for the attack algorithm.
                      Smaller values produce better results but are
                      slower to converge.
    :return: a tensor for the adversarial example.
    """
    if clip_min is not None:
        assert np.all(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        assert np.all(tf.math.less_equal(x, clip_max))

    y, nb_classes = get_or_guess_labels(model_fn, x, y=y, targeted=targeted)
    shape = x.shape

    # cast to tensor if provided as numpy array
    original_x = tf.cast(x, tf.float32)
    assert y.shape.as_list() == original_x.shape.as_list()

    # re-scale x to [0, 1]
    x = original_x
    x = (x - clip_min) / (clip_max - clip_min)
    x = tf.clip_by_value(x, 0., 1.)

    # scale to [-1, 1]
    x = (x * 2.) - 1.

    # convert tonh-space
    x = tf.atanh(x * .999999)

    # parameters for the binary search
    # const is the tradeoff constant (c in the paper)
    lower_bound = tf.zeros(shape)
    upper_bound = tf.ones(shape) * 1e10
    const = tf.ones(shape) * initial_const

    # placeholder variables for best values
    best_l2 = tf.fill(shape, 1e10)
    best_score = tf.fill(shape, -1.)
    best_attack = original_x

    # convience function for 'clipping' to input range
    def clip_tanh(x):
        return (tf.tanh(x) + 1.) / 2 * (clip_max - clip_min) + clip_max

    # convience function for comparing
    def compare(x, y):
        # TODO: check if consistens with tf1 version
        if targeted:
            return x == y
        else:
            return x != y

    # setup loss function
    @tf.function
    def loss_fn(x, x_new, y_true, y_pred):
        other = clip_tanh(x)
        l2_dist = tf.reduce_sum(
            tf.square(x_new - other), list(range(1, len(shape))))

        real = tf.reduce_sum(y_true * y_pred, 1)
        other = tf.reduce_max(
            (1. - y_true) * y_pred - y_true * 10_000, 1)

        if targeted:
            # if targeted, optimize for making the other class most likely
            loss_1 = tf.maximum(0, other - real + confidence)
        else:
            # if untargeted, optimize for making this class least likely.
            loss_1 = tf.maximum(0, real - other + confidence)

        # sum up losses
        loss_2 = tf.reduce_sum(l2_dist)
        loss_1 = tf.reduce_sum(const * loss_1)
        loss = loss_1 + loss_2
        return loss, l2_dist

    for outer_step in range(binary_search_steps):
        # reset Adam
        optimzer = tf.keras.optimzers.Adam(learning_rate)

        # variables to keep track in the inner loop
        current_best_l2 = tf.fill(shape, 1e10)
        current_best_score = tf.fill(shape, -1.)

        # The last iteration (if we run many steps) repeat the search once.
        if binary_search_steps >= 10 and outer_step == binary_search_steps - 1:
            const = upper_bound

        # early stopping criteria
        prev = 1e6

        # temporary variables + the perturbation
        modifier = tf.zeros(shape, dtype=x.dtype)
        x_tmp = x
        y_tmp = y

        for iteration in range(max_iterations):
            # compute the actual attack
            x_new = clip_tanh(modifier + x_tmp)

            with tf.GradientTape() as tape:
                tape.watch(x_new)
                preds = model_fn(x_new)
                loss, l2_dist = loss_fn(
                    x=x_tmp, x_new=x_new, y_true=y, y_pred=preds)

            optimzer.minimize(loss, [modifier])

            # check if we made progress, abort otherwise
            if abort_early and iteration % ((max_iterations // 10) or 1) == 0:
                if loss > prev * 0.9999:
                    # Failed to make progress; stop early
                    break

                prev = loss

            # TODO: vectorize
            for e, (l2, pred, ii) in enumerate(l2_dist, preds, x_new):
                lab = tf.argmax(y[e])

                # update optimization tracking variables
                if l2 < current_best_l2[e] and compare(pred, lab):
                    current_best_l2[e] = l2
                    current_best_score[e] = tf.argmax[pred]

                # update binary search tracking variables
                if l2 < best_l2[e] and compare(pred, lab):
                    best_l2[e] = l2
                    best_score[e] = tf.argmax(pred)
                    best_attack[e] = ii

        # adjust binary search parameters
        # TODO: vectorize
        for e in range(shape[-1]):
            # TODO: rework for tf2
            if compare(best_score[e], tf.argmax(y[e])) and best_score[e] != -1:
                # success, divide const by two
                upper_bound[e] = min(upper_bound[e], const[e])
                if upper_bound[e] < 1e9:
                    const[e] = (lower_bound[e] + upper_bound[e]) / 2.

            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                lower_bound[e] = max(lower_bound[e], const[e])
                if upper_bound[e] < 1e9:
                    const[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    const[e] *= 10

        return best_attack
