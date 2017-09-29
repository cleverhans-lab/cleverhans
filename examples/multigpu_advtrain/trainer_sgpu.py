
import tensorflow as tf
from tensorflow.python.client import timeline


class TrainerSingleGPU(TrainManager):
    def __init__(self, **kwargs):
        super(TrainerSingleGPU, self).__init__(**kwargs)
        self.manager = manager
        self.hparams = manager.hparams
        self.sess = manager.sess
        self.step_num = 0

        self.create_train_graph()

    def is_finished(self):
        """
        Single GPU trainer has no cache.
        """
        return True

    def set_input(self, X_batch=None, Y_batch=None):
        x_pre, x, y = self.manager.g0_inputs
        fd[x_pre] = X_batch
        fd[y] = Y_batch
        fetches = self.outputs[0]
        return fetches, fd

    def proc_fvals(self, fvals):
        """
        Nothing to post-process on single GPU.
        """
        return True

    def sync_params(self, forced=False):
        """
        Nothing to sync on single GPU.
        """
        return True

    def create_train_graph(self):
        hparams = self.hparams
        model = self.model
        x_pre, x, y = self.g0_inputs
        sess = self.sess

        if not hparams.adv_train:
            logging.info("Adversarial training")

            logging.info("Initializing train attack %s" %
                         hparams.attack_type_train)
            preds_2_adv = model.get_probs(adv2_x, training=True,
                                          bn_training=False)

        if not hparams.adv_train:
            preds = model.get_probs(x2, training=True,
                                    bn_training=True)
            preds_2_adv = None
        else:
            adv2_x = create_adv_by_name(
                model, x, hparams.attack_type_train, sess,
                y=y, nb_iter=hparams.attack_nb_iter_train,
                dataset=hparams.dataset)
            if hparams.only_adv_train:
                preds = None
                preds_2_adv = model.get_probs(adv2_x, training=True,
                                            bn_training=True)
            else:
                preds = model.get_probs(x2, training=True)
                preds_2_adv = model.get_probs(adv2_x, training=True,
                                            bn_training=True)
        train_fetches = self.build_train_op(preds, y2, preds_2_adv)

        self.inputs = [self.g0_inputs]
        self.outputs = [train_fetches]
