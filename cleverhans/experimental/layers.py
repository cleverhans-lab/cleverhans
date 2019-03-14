from cleverhans.picklable_model import Layer
import tensorflow as tf

NCHW, NHWC = 'NCHW NHWC'.split()


class Downscale2D(Layer):
  """Box downscaling.

  :param n: integer scale.
  :param order: enum(NHCW, NHWC), the oder of channesl vs dimensions.
  """

  def __init__(self, n=2, order=NHWC, **kwargs):
    super(Downscale2D, self).__init__(**kwargs)
    self.n = n
    self.order = order

  def set_input_shape(self, shape):
    self.input_shape = shape
    if self.order == NHWC:
      batch_size, rows, cols, channels = shape
    else:
      batch_size, channels, rows, cols = shape
    rows = rows // self.n
    cols = cols // self.n
    if self.order == NHWC:
      shape = (batch_size, rows, cols, channels)
    else:
      shape = (batch_size, channels, rows, cols)
    self.output_shape = shape

  def fprop(self, x):
    return downscale2d(x)

  def get_params(self):
    return []


def downscale2d(x, n=2, order=NHWC):
  """Box downscaling.

  Args:
    x: 4D tensor in order format.
    n: integer scale.
    order: enum(NCHW, NHWC), the order of channels vs dimensions.

  Returns:
    4D tensor down scaled by a factor n.

  Raises:
    ValueError: if order not NCHW or NHWC.
  """
  check_order(order)
  if n <= 1:
    return x
  if order == NCHW:
    pool2, pooln = [1, 1, 2, 2], [1, 1, n, n]
  else:
    pool2, pooln = [1, 2, 2, 1], [1, n, n, 1]
  if n % 2 == 0:
    x = tf.nn.avg_pool(x, pool2, pool2, 'VALID', order)
    return downscale2d(x, n // 2, order)
  return tf.nn.avg_pool(x, pooln, pooln, 'VALID', order)


def check_order(order):
  if order not in (NCHW, NHWC):
    raise ValueError('Unsupported tensor order %s' % order)


class NormalizeRMS(Layer):

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape

  def fprop(self, x, **kwargs):
    out = x * tf.rsqrt(tf.reduce_mean(tf.square(x),
                                      axis=1, keep_dims=True) + 1e-8)
    return out

  def get_params(self):
    return []


class MaxOut(Layer):

  def __init__(self, num_pieces=2, **kwargs):
    super(MaxOut, self).__init__(**kwargs)
    self.num_pieces = num_pieces

  def get_params(self):
    return []

  def set_input_shape(self, shape):
    self.input_shape = shape
    b, r, c, ch = shape
    assert ch % self.num_pieces == 0
    shape = (b, r, c, ch // self.num_pieces)
    self.output_shape = shape
    self.expanded_shape = shape + (self.num_pieces,)
    self.expanded_shape = list(self.expanded_shape)
    self.expanded_shape[0] = -1

  def fprop(self, x, **kwargs):
    x = tf.reshape(x, self.expanded_shape)
    x = tf.reduce_max(x, axis=4)
    return x
