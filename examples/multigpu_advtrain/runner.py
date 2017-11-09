import tensorflow as tf
import numpy as np


class RunnerMultiGPU(object):
    def __init__(self, sess, input_shape):
        self.sess = sess
        self.input_shape = input_shape
        self.feed_dict = {}
        self.init_inputs()

    def init_inputs(self):
        input_shape = self.input_shape
        # Define input TF placeholder
        with tf.device('/gpu:0'):
            x = tf.placeholder(tf.float32, shape=input_shape, name='x')
            y = tf.placeholder(tf.float32, shape=(input_shape[0], 10),
                               name='y')
        self.g0_inputs = (x, x, y)

    def set_input(self, X_batch=None):
        inputs = self.inputs
        outputs = self.outputs

        # data for first gpu
        fd = {}
        if X_batch is not None:
            _, x, y = self.g0_inputs
            fd[x] = X_batch
            fd[y] = np.zeros((X_batch.shape[0], 10))
            self.next_vals[0] = (X_batch,)
        else:
            self.next_vals[0] = None

        # set feed_dict for each gpu if there is something to run for that gpu
        # collect outputs to be fetched
        fetches = []
        self.active_gpus = []
        for i in range(len(outputs)):
            if self.next_vals[i] is None:
                self.active_gpus += [False]
                continue
            self.active_gpus += [True]
            if i > 0:
                for j in range(len(inputs[i])):
                    fd[inputs[i][j]] = self.next_vals[i][j]
            for j in range(len(outputs[i])):
                fetches += [outputs[i][j]]

        fd.update(self.feed_dict)

        return fetches, fd

    def proc_fvals(self, fvals):
        inputs = self.inputs
        outputs = self.outputs

        # move data for next step
        cur = 0
        for i in range(len(inputs)-1):
            if not self.active_gpus[i]:
                self.next_vals[i+1] = None
                continue
            self.next_vals[i+1] = []
            for j in range(len(outputs[i])):
                self.next_vals[i+1] += [fvals[cur]]
                cur += 1
            if i == 0:
                self.next_vals[0] = None

        last_fvals = []
        while cur < len(fvals):
            last_fvals += [fvals[cur]]
            cur += 1
        return last_fvals

    def is_finished(self):
        return all(v is None for v in self.next_vals)

    def run_simple(self, X_batch=None):
        fetches, feed_dict = self.set_input(X_batch)
        fvals = self.sess.run(fetches, feed_dict=feed_dict)
        return self.proc_fvals(fvals)

    def run(self, X_batch=None):
        return self.run_simple(X_batch)
