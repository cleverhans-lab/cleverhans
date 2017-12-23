import numpy as np
from cleverhans.utils_tf import clip_eta
from cleverhans.devtools.checks import CleverHansTest
import tensorflow as tf


class TestUtils(CleverHansTest):

    def setUp(self):
        super(TestUtils, self).setUp()

        self.sess = tf.Session()

    def test_clip_eta_norm_0(self):
        # Test that `clip_eta` still works when the norm of `eta` is
        # zero. This used to cause a divide by zero for ord 1 and ord
        # 2.
        eta = tf.zeros((5, 3))
        assert eta.dtype == tf.float32, eta.dtype
        eps = .25
        for ord_arg in [np.inf, 1, 2]:
            clipped = clip_eta(eta, ord_arg, eps)
            clipped = self.sess.run(clipped)
            assert not np.any(np.isinf(clipped))
            assert not np.any(np.isnan(clipped)), (ord_arg, clipped)

    def test_clip_eta_goldilocks(self):
        # Test that the clipping handles perturbations that are
        # too small, just right, and too big correctly
        eta = tf.constant([[2.], [3.], [4.]])
        assert eta.dtype == tf.float32, eta.dtype
        eps = 3.
        for ord_arg in [np.inf, 1, 2]:
            for sign in [-1., 1.]:
                clipped = clip_eta(eta * sign, ord_arg, eps)
                clipped_value = self.sess.run(clipped)
                gold = sign * np.array([[2.], [3.], [3.]])
                self.assertClose(clipped_value, gold)
                grad, = tf.gradients(clipped, eta)
                grad_value = self.sess.run(grad)
                # Note: the second 1. is debatable (the left-sided derivative
                # and the right-sided derivative do not match, so formally
                # the derivative is not defined). This test makes sure that
                # we at least handle this oddity consistently across all the
                # argument values we test
                gold = sign * np.array([[1.], [1.], [0.]])
                assert np.allclose(grad_value, gold)
