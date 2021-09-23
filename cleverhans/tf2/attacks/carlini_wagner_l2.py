"""The CarliniWagnerL2 attack.
"""
import numpy as np
import tensorflow as tf
from cleverhans.tf2.utils import get_or_guess_labels, set_with_mask


def carlini_wagner_l2(model_fn, x, **kwargs):
    """
    This is the function interface for the Carlini-Wagner-L2 attack.
    For more details on the attack and the parameters see the corresponding class.
    """
    return CarliniWagnerL2(model_fn, **kwargs).attack(x)


class CarliniWagnerL2Exception(Exception):
    pass


class CarliniWagnerL2(object):
    def __init__(
        self,
        model_fn,
        y=None,
        targeted=False,
        batch_size=128,
        clip_min=0.0,
        clip_max=1.0,
        binary_search_steps=5,
        max_iterations=1_000,
        abort_early=True,
        confidence=0.0,
        initial_const=1e-2,
        learning_rate=5e-3,
    ):
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
        :param y: (optional) Tensor with target labels.
        :param targeted: (optional) Targeted attack?
        :param batch_size (optional): Number of attacks to run simultaneously.
        :param clip_min: (optional) float. Minimum float values for adversarial example components.
        :param clip_max: (optional) float. Maximum float value for adversarial example components.
        :param binary_search_steps (optional): The number of times we perform binary
                                search to find the optimal tradeoff-
                                constant between norm of the purturbation
                                and confidence of the classification.
        :param max_iterations (optional): The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early (optional): If true, allows early aborts if gradient descent
                        is unable to make progress (i.e., gets stuck in
                        a local minimum).
        :param confidence (optional): Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param initial_const (optional): The initial tradeoff-constant used to tune the
                          relative importance of the size of the perturbation
                          and confidence of classification.
                          If binary_search_steps is large, the initial
                          constant is not important. A smaller value of
                          this constant gives lower distortion results.
        :param learning_rate (optional): The learning rate for the attack algorithm.
                          Smaller values produce better results but are
                          slower to converge.
        """
        self.model_fn = model_fn

        self.batch_size = batch_size

        self.y = y
        self.targeted = y is not None

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.learning_rate = learning_rate

        self.confidence = confidence
        self.initial_const = initial_const

        # the optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        super(CarliniWagnerL2, self).__init__()

    def attack(self, x):
        """
        Returns adversarial examples for the tensor.
        :param x: input tensor.
        :return: a numpy tensor with the adversarial example.
        """
        adv_ex = np.zeros_like(x)
        for i in range(0, len(x), self.batch_size):
            adv_ex[i : i + self.batch_size] = self._attack(
                x[i : i + self.batch_size]
            ).numpy()

        return adv_ex

    def _attack(self, x):
        if self.clip_min is not None:
            if not np.all(tf.math.greater_equal(x, self.clip_min)):
                raise CarliniWagnerL2Exception(
                    f"The input is smaller than the minimum value of {self.clip_min}r"
                )

        if self.clip_max is not None:
            if not np.all(tf.math.less_equal(x, self.clip_max)):
                raise CarliniWagnerL2Exception(
                    f"The input is greater than the maximum value of {self.clip_max}!"
                )

        # cast to tensor if provided as numpy array
        original_x = tf.cast(x, tf.float32)
        shape = original_x.shape

        y, _ = get_or_guess_labels(
            self.model_fn, original_x, y=self.y, targeted=self.targeted
        )

        if not y.shape.as_list()[0] == original_x.shape.as_list()[0]:
            raise CarliniWagnerL2Exception("x and y do not have the same shape!")

        # re-scale x to [0, 1]
        x = original_x
        x = (x - self.clip_min) / (self.clip_max - self.clip_min)
        x = tf.clip_by_value(x, 0.0, 1.0)

        # scale to [-1, 1]
        x = (x * 2.0) - 1.0

        # convert tonh-space
        x = tf.atanh(x * 0.999999)

        # parameters for the binary search
        lower_bound = tf.zeros(shape[:1])
        upper_bound = tf.ones(shape[:1]) * 1e10

        const = tf.ones(shape) * self.initial_const

        # placeholder variables for best values
        best_l2 = tf.fill(shape[:1], 1e10)
        best_score = tf.fill(shape[:1], -1)
        best_score = tf.cast(best_score, tf.int32)
        best_attack = original_x

        # convience function for comparing
        compare_fn = tf.equal if self.targeted else tf.not_equal

        # the perturbation
        modifier = tf.Variable(tf.zeros(shape, dtype=x.dtype), trainable=True)

        for outer_step in range(self.binary_search_steps):
            # at each iteration reset variable state
            modifier.assign(tf.zeros(shape, dtype=x.dtype))
            for var in self.optimizer.variables():
                var.assign(tf.zeros(var.shape, dtype=var.dtype))

            # variables to keep track in the inner loop
            current_best_l2 = tf.fill(shape[:1], 1e10)
            current_best_score = tf.fill(shape[:1], -1)
            current_best_score = tf.cast(current_best_score, tf.int32)

            # The last iteration (if we run many steps) repeat the search once.
            if (
                self.binary_search_steps >= 10
                and outer_step == self.binary_search_steps - 1
            ):
                const = upper_bound

            # early stopping criteria
            prev = None

            for iteration in range(self.max_iterations):
                x_new, loss, preds, l2_dist = self.attack_step(x, y, modifier, const)

                # check if we made progress, abort otherwise
                if (
                    self.abort_early
                    and iteration % ((self.max_iterations // 10) or 1) == 0
                ):
                    if prev is not None and loss > prev * 0.9999:
                        break

                    prev = loss

                lab = tf.argmax(y, axis=1)

                pred_with_conf = (
                    preds - self.confidence
                    if self.targeted
                    else preds + self.confidence
                )
                pred_with_conf = tf.argmax(pred_with_conf, axis=1)

                pred = tf.argmax(preds, axis=1)
                pred = tf.cast(pred, tf.int32)

                # compute a binary mask of the tensors we want to assign
                mask = tf.math.logical_and(
                    tf.less(l2_dist, current_best_l2), compare_fn(pred_with_conf, lab)
                )

                # all entries which evaluate to True get reassigned
                current_best_l2 = set_with_mask(current_best_l2, l2_dist, mask)
                current_best_score = set_with_mask(current_best_score, pred, mask)

                # if the l2 distance is better than the one found before
                # and if the example is a correct example (with regards to the labels)
                mask = tf.math.logical_and(
                    tf.less(l2_dist, best_l2), compare_fn(pred_with_conf, lab)
                )

                best_l2 = set_with_mask(best_l2, l2_dist, mask)
                best_score = set_with_mask(best_score, pred, mask)

                # mask is of shape [batch_size]; best_attack is [batch_size, image_size]
                # need to expand
                mask = tf.reshape(mask, [-1, 1, 1, 1])
                mask = tf.tile(mask, [1, *best_attack.shape[1:]])

                best_attack = set_with_mask(best_attack, x_new, mask)

            # adjust binary search parameters
            lab = tf.argmax(y, axis=1)
            lab = tf.cast(lab, tf.int32)

            # we first compute the mask for the upper bound
            upper_mask = tf.math.logical_and(
                compare_fn(best_score, lab),
                tf.not_equal(best_score, -1),
            )
            upper_bound = set_with_mask(
                upper_bound, tf.math.minimum(upper_bound, const), upper_mask
            )

            # based on this mask compute const mask
            const_mask = tf.math.logical_and(
                upper_mask,
                tf.less(upper_bound, 1e9),
            )
            const = set_with_mask(const, (lower_bound + upper_bound) / 2.0, const_mask)

            # else case is the negation of the inital mask
            lower_mask = tf.math.logical_not(upper_mask)
            lower_bound = set_with_mask(
                lower_bound, tf.math.maximum(lower_bound, const), lower_mask
            )

            const_mask = tf.math.logical_and(
                lower_mask,
                tf.less(upper_bound, 1e9),
            )
            const = set_with_mask(const, (lower_bound + upper_bound) / 2, const_mask)

            const_mask = tf.math.logical_not(const_mask)
            const = set_with_mask(const, const * 10, const_mask)

        return best_attack

    def attack_step(self, x, y, modifier, const):
        x_new, grads, loss, preds, l2_dist = self.gradient(x, y, modifier, const)

        self.optimizer.apply_gradients([(grads, modifier)])
        return x_new, loss, preds, l2_dist

    @tf.function
    def gradient(self, x, y, modifier, const):
        # compute the actual attack
        with tf.GradientTape() as tape:
            adv_image = modifier + x
            x_new = clip_tanh(adv_image, clip_min=self.clip_min, clip_max=self.clip_max)
            preds = self.model_fn(x_new)
            loss, l2_dist = loss_fn(
                x=x,
                x_new=x_new,
                y_true=y,
                y_pred=preds,
                confidence=self.confidence,
                const=const,
                targeted=self.targeted,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )

        grads = tape.gradient(loss, adv_image)
        return x_new, grads, loss, preds, l2_dist


def l2(x, y):
    # technically squarred l2
    return tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape))))


def loss_fn(
    x,
    x_new,
    y_true,
    y_pred,
    confidence,
    const=0,
    targeted=False,
    clip_min=0,
    clip_max=1,
):
    other = clip_tanh(x, clip_min=clip_min, clip_max=clip_max)
    l2_dist = l2(x_new, other)

    real = tf.reduce_sum(y_true * y_pred, 1)
    other = tf.reduce_max((1.0 - y_true) * y_pred - y_true * 10_000, 1)

    if targeted:
        # if targeted, optimize for making the other class most likely
        loss_1 = tf.maximum(0.0, other - real + confidence)
    else:
        # if untargeted, optimize for making this class least likely.
        loss_1 = tf.maximum(0.0, real - other + confidence)

    # sum up losses
    loss_2 = tf.reduce_sum(l2_dist)
    loss_1 = tf.reduce_sum(const * loss_1)
    loss = loss_1 + loss_2
    return loss, l2_dist


def clip_tanh(x, clip_min, clip_max):
    return ((tf.tanh(x) + 1) / 2) * (clip_max - clip_min) + clip_min
