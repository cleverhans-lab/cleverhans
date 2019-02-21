"""
Initializers.
"""

import tensorflow as tf


class HeReLuNormalInitializer(tf.initializers.random_normal):
  """
  The initializer from He et al 2015
  """
  def __init__(self, dtype=tf.float32):
    super(HeReLuNormalInitializer, self).__init__(dtype=dtype)

  def get_config(self):
    return dict(dtype=self.dtype.name)

  def __call__(self, shape, dtype=None, partition_info=None):
    del partition_info
    dtype = self.dtype if dtype is None else dtype
    std = tf.rsqrt(tf.cast(tf.reduce_prod(shape[:-1]), tf.float32) + 1e-7)
    return tf.random_normal(shape, stddev=std, dtype=dtype)
