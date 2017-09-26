import tensorflow as tf
from tensorflow.python.client import timeline


class RunnerMultiGPU(object):
    def __init__(self, inputs, outputs, sync_ops, manager):
        self.manager = manager
        self.hparams = manager.hparams
        self.sess = manager.sess
        self.inputs = inputs
        self.outputs = outputs
        self.sync_ops = sync_ops
        self.feed_dict = {}
        self.step_num = 0
        self.next_vals = [None] * len(self.inputs)

    def init_tf(self, X_batch, Y_batch):
        x_pre, x, y = self.manager.g0_inputs
        fd = {x_pre: X_batch, y: Y_batch}
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op, feed_dict=fd)

    def is_finished(self):
        return self.next_vals[-1] is None

    def set_input(self, X_batch=None, Y_batch=None):
        inputs = self.inputs
        outputs = self.outputs

        # data for first gpu
        fd = {}
        if X_batch is not None:
            x_pre, x, y = self.manager.g0_inputs
            fd[x_pre] = X_batch
            fd[y] = Y_batch
            self.next_vals[0] = (X_batch, Y_batch)
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

    def run(self, X_batch=None, Y_batch=None):
        if self.step_num == len(self.inputs)+1:
            self.run_with_graph(X_batch, Y_batch)
        else:
            self.run_simple(X_batch, Y_batch)

    def run_simple(self, X_batch=None, Y_batch=None):
        fetches, feed_dict = self.set_input(X_batch, Y_batch)
        fvals = self.sess.run(fetches, feed_dict=feed_dict)
        self.proc_fvals(fvals)
        self.step_num += 1

    def run_with_graph(self, X_batch, Y_batch):
        manager = self.manager
        fetches, feed_dict = self.set_input(X_batch, Y_batch)

        manager.writer.add_graph(self.sess.graph)
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        fvals = self.sess.run(fetches,
                              feed_dict=feed_dict,
                              options=run_options,
                              run_metadata=run_metadata)
        manager.writer.add_run_metadata(run_metadata, 'graph')

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

        self.proc_fvals(fvals)
        self.step_num += 1

    def sync_params(self, forced=False):
        if forced or (self.step_num % self.hparams.sync_step == 0):
            self.sess.run(self.sync_ops)
