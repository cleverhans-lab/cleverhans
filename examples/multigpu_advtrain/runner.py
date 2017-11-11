from collections import OrderedDict


class RunnerMultiGPU(object):
    def __init__(self, sess, inputs, outputs):
        self.sess = sess
        self.inputs = inputs
        self.outputs = outputs
        self.next_vals = [None] * len(inputs)
        self.feed_dict = {}
        self.step_num = 0

    def set_input(self, X_batch=None):
        inputs = self.inputs
        outputs = self.outputs

        # data for first gpu
        fd = {}
        if X_batch is not None:
            self.next_vals[0] = []
            for i, vname in enumerate(self.inputs[0]):
                if vname in X_batch:
                    self.next_vals[0] += [X_batch[vname]]
                else:
                    self.next_vals[0] += [None]
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
            for j, k in enumerate(inputs[i]):
                if self.next_vals[i][j] is not None:
                    fd[inputs[i][k]] = self.next_vals[i][j]
            for k, v in outputs[i].iteritems():
                fetches += [v]

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

        last_fvals = OrderedDict()
        if self.active_gpus[-1]:
            assert cur+len(outputs[-1]) == len(fvals)
            for k, v in outputs[-1].iteritems():
                last_fvals[k] = fvals[cur]
                cur += 1
        return last_fvals

    def is_finished(self):
        return all(v is None for v in self.next_vals)

    def run_simple(self, X_batch=None):
        fetches, feed_dict = self.set_input(X_batch)
        fvals = self.sess.run(fetches, feed_dict=feed_dict)
        return self.proc_fvals(fvals)

    def run(self, X_batch=None):
        last_fvals = self.run_simple(X_batch)
        self.step_num += 1
        return last_fvals
