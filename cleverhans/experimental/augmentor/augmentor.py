from cleverhans.model import Model
import tensorflow as tf
from tensorflow.python.framework import function


class Augmentor(Model):

  def __init__(self, raw):
    self.raw = raw

  def get_params(self):
    return self.raw.get_params()

  def fprop(self, x):
    mode = "REFLECT"
    assert mode in 'REFLECT SYMMETRIC CONSTANT'.split()
    pad = [2, 2]

    def _pad(img):
      return tf.pad(img, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode)
    xp = tf.map_fn(_pad, x)
    xs = []
    for i in xrange(pad[0] * 2):
      for j in xrange(pad[1] * 2):
        xs.append(tf.slice(xp, [0, i, j, 0], tf.shape(x)))
        with tf.device("/CPU:0"):
          xs.append(tf.image.flip_left_right(xs[-1]))

    @function.Defun(tf.float32)
    def f(xarg):
      xarg.set_shape(x.get_shape())
      return self.raw.get_logits(xarg)

    logits = [f(e) for e in xs]
    logits = tf.add_n(logits) / len(logits)
    return {'logits': logits}

  def get_dataset_factory(self):
    return self.raw.dataset_factory

  dataset_factory = property(get_dataset_factory)

  def make_input_placeholder(self):
    return self.raw.make_input_placeholder()

  def make_label_placeholder(self):
    return self.raw.make_label_placeholder()


  def no_defun_fprop(self, x):
    mode = "REFLECT"
    assert mode in 'REFLECT SYMMETRIC CONSTANT'.split()
    pad = [2, 2]

    def _pad(img):
      return tf.pad(img, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode)
    xp = tf.map_fn(_pad, x)
    xs = []
    for i in xrange(pad[0] * 2):
      for j in xrange(pad[1] * 2):
        xs.append(tf.slice(xp, [0, i, j, 0], tf.shape(x)))
        xs.append(tf.image.flip_left_right(xs[-1]))
        #xs[-1] = tf.Print(xs[-1], [xs[-1][0, 0, 0, 0]], message="corner " + str(i) + " " + str(j))
    logits = [self.raw.get_logits(
        e, cifar10_model_hid_keep_prob=1.) for e in xs]
    #logits = [tf.Print(e, [e[0, :]], message="logits_" + str(i), summarize=10) for i, e in enumerate(logits)]
    logits = tf.add_n(logits) / len(logits)
    return {'logits': logits}
