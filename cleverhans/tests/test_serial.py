import tensorflow as tf
import numpy as np
from cleverhans.serial import PicklableVariable
from cleverhans.serial import load
from cleverhans.serial import save


def test_save_and_load_var():
    """
    Tests that we can save and load a PicklableVariable with joblib
    """
    sess = tf.Session()
    with sess.as_default():
        x = np.ones(1)
        xv = PicklableVariable(x)
        xv.var.initializer.run()
        save("/tmp/var.joblib", xv)
        sess.run(tf.assign(xv.var, np.ones(1) * 2))
        new_xv = load("/tmp/var.joblib")
        assert np.allclose(sess.run(xv.var), np.ones(1) * 2)
        assert np.allclose(sess.run(new_xv.var), np.ones(1))
