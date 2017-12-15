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
        ord_arg = 2
        eps = .25
        for ord_arg in [np.inf, 1, 2]:
            clipped = clip_eta(eta, ord_arg, eps)
            clipped = self.sess.run(clipped)
            assert not np.any(np.isinf(clipped))
            assert not np.any(np.isnan(clipped)), (ord_arg, clipped)

