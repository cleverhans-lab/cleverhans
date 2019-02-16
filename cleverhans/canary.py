"""
Canary code that dies if the underlying hardware / drivers aren't working right.
"""
import time

import numpy as np
import tensorflow as tf
from cleverhans.utils_tf import infer_devices

last_run = None


def run_canary():
  """
  Runs some code that will crash if the GPUs / GPU driver are suffering from
  a common bug. This helps to prevent contaminating results in the rest of
  the library with incorrect calculations.
  """

  # Note: please do not edit this function unless you have access to a machine
  # with GPUs suffering from the bug and can verify that the canary still
  # crashes after your edits. Due to the transient nature of the GPU bug it is
  # not possible to unit test the canary in our continuous integration system.

  global last_run
  current = time.time()
  if last_run is None or current - last_run > 3600:
    last_run = current
  else:
    # Run the canary at most once per hour
    return

  # Try very hard not to let the canary affect the graph for the rest of the
  # python process
  canary_graph = tf.Graph()
  with canary_graph.as_default():
    devices = infer_devices()
    num_devices = len(devices)
    if num_devices < 3:
      # We have never observed GPU failure when less than 3 GPUs were used
      return

    v = np.random.RandomState([2018, 10, 16]).randn(2, 2)
    # Try very hard not to let this Variable end up in any collections used
    # by the rest of the python process
    w = tf.Variable(v, trainable=False, collections=[])
    loss = tf.reduce_sum(tf.square(w))

    grads = []
    for device in devices:
      with tf.device(device):
        grad, = tf.gradients(loss, w)
        grads.append(grad)

    sess = tf.Session()
    sess.run(tf.variables_initializer([w]))
    grads = sess.run(grads)
    first = grads[0]
    for grad in grads[1:]:
      if not np.allclose(first, grad):
        first_string = str(first)
        grad_string = str(grad)
        raise RuntimeError("Something is wrong with your GPUs or GPU driver."
                           "%(num_devices)d different GPUS were asked to "
                           "calculate the same 2x2 gradient. One returned "
                           "%(first_string)s and another returned "
                           "%(grad_string)s. This can usually be fixed by "
                           "rebooting the machine." %
                           {"num_devices" : num_devices,
                            "first_string" : first_string,
                            "grad_string" : grad_string})
    sess.close()

if __name__ == "__main__":
  run_canary()
