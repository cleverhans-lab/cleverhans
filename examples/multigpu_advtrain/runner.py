import tensorflow as tf
from tensorflow.python.client import timeline


class TrainerMultiGPU(object):
    def __init__(self, **kwargs):
        super(TrainerMultiGPU, self).__init__(**kwargs)
        self.manager = manager
        self.hparams = manager.hparams
        self.sess = manager.sess
        self.feed_dict = {}
        self.step_num = 0

        self.create_train_graph()
        self.next_vals = [None] * len(self.inputs)

    def create_train_graph(self):
        assert '_multigpu' in self.hparams.attack_type_train

        hparams = self.hparams
        model = self.model
        x_pre, x, y = self.g0_inputs
        sess = self.sess

        # Generates steps on gpus 0-(ngpu-1)
        logging.info("Initializing train attack %s" %
                     hparams.attack_type_train)
        inputs, outputs = create_adv_by_name(
            model, x, hparams.attack_type_train,
            sess, y=y, nb_iter=hparams.attack_nb_iter_train,
            dataset=hparams.dataset, ngpu=hparams.ngpu)

        assert len(inputs) == len(outputs)
        # 0
        # inputs[0] = (x_pre, y)

        # copy y forward
        for i in range(len(outputs)):
            if i > 0:
                with tf.device(inputs[i][-1].device):
                    y2 = clone_variable('y%d' % i, y)
            else:
                y2 = y
            inputs[i] = inputs[i] + (y2,)
            outputs[i] = outputs[i] + (y2,)

        # train step on last gpu
        x, adv_x, y = outputs[-1]
        device_name = '/gpu:%d' % (hparams.ngpu-1)
        model.set_device(device_name)
        with tf.device(device_name):
            with tf.variable_scope('last'):
                x2 = clone_variable('x_-1', x)
                adv2_x = clone_variable('adv_x_-1', adv_x)
                y2 = clone_variable('y_-1', y)
                inputs += [(x2, adv2_x, y2)]
                if not hparams.adv_train:
                    preds = model.get_probs(x2, training=True,
                                            bn_training=True)
                    preds_2_adv = None
                elif not hparams.only_adv_train:
                    preds = model.get_probs(x2, training=True)
                    preds_2_adv = model.get_probs(adv2_x, training=True,
                                                  bn_training=True)
                else:
                    preds = None
                    preds_2_adv = model.get_probs(adv2_x, training=True,
                                                  bn_training=True)
                train_fetches = self.build_train_op(preds, y2, preds_2_adv)

        outputs += [train_fetches]

        device_name = '/gpu:%d' % (hparams.ngpu-1)
        model.set_device(device_name)
        with tf.device(device_name):
            sync_ops = model.create_sync_ops(host_device=device_name)

        self.inputs = inputs
        self.outputs = outputs
        self.sync_ops = sync_ops

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

    def sync_params(self, forced=False):
        if forced or (self.step_num % self.hparams.sync_step == 0):
            self.sess.run(self.sync_ops)

    def is_finished(self):
        return self.next_vals[-1] is None
