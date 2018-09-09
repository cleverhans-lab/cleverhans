from cleverhans.picklable_model import MLP, Dropout
from six.moves import range
import tensorflow as tf


def test_no_drop():
    # Make sure dropout does nothing by default (so it does not cause
    # stochasticity at test time)

    model = MLP(input_shape=[1, 1], layers=[Dropout(name='output')])
    x = tf.constant([[1]], dtype=tf.float32)
    y = model.get_layer(x, 'output')
    sess = tf.Session()
    # Do multiple runs because dropout is stochastic
    for i in range(10):
        y_value = sess.run(y)
        assert y_value == 1.


def test_drop():
    # Make sure dropout is activated successfully

    # We would like to configure the test to deterministically drop,
    # so that the test does not need to use multiple runs.
    # However, tf.nn.dropout divides by include_prob, so zero or
    # infinitesimal include_prob causes NaNs.
    # 1e-8 does not cause NaNs and shouldn't be a significant source
    # of test flakiness relative to dependency downloads failing, etc.
    model = MLP(input_shape=[1, 1], layers=[Dropout(name='output',
                                                    include_prob=1e-8)])
    x = tf.constant([[1]], dtype=tf.float32)
    y = model.get_layer(x, 'output', dropout=True)
    sess = tf.Session()
    y_value = sess.run(y)
    # Subject to very rare random failure because include_prob is not exact 0
    assert y_value == 0., y_value


def test_override():
    # Make sure dropout_dict changes dropout probabilities successful

    # We would like to configure the test to deterministically drop,
    # so that the test does not need to use multiple runs.
    # However, tf.nn.dropout divides by include_prob, so zero or
    # infinitesimal include_prob causes NaNs.
    # For this test, random failure to drop will not cause the test to fail.
    # The stochastic version should not even run if everything is working
    # right.
    model = MLP(input_shape=[1, 1], layers=[Dropout(name='output',
                                                    include_prob=1e-8)])
    x = tf.constant([[1]], dtype=tf.float32)
    dropout_dict = {'output': 1.}
    y = model.get_layer(x, 'output', dropout=True, dropout_dict=dropout_dict)
    sess = tf.Session()
    y_value = sess.run(y)
    assert y_value == 1., y_value
