import logging

from evaluator import create_adv_by_name
from trainer import TrainManager


class TrainerSingleGPU(TrainManager):
    def __init__(self, *args, **kwargs):
        super(TrainerSingleGPU, self).__init__(*args, **kwargs)
        self.create_train_graph()

    def create_train_graph(self):
        hparams = self.hparams
        model = self.model
        x_pre, x, y = self.g0_inputs
        sess = self.sess

        if not hparams.adv_train:
            logging.info("Naive training")

            model.set_training(training=True, bn_training=True)
            preds = model.get_probs(x)
            preds_adv = None
        else:
            logging.info("Adversarial training")
            logging.info("Initializing train attack %s" %
                         hparams.attack_type_train)

            adv_x = create_adv_by_name(
                model, x, hparams.attack_type_train, sess,
                y=y, nb_iter=hparams.attack_nb_iter_train,
                dataset=hparams.dataset)
            if hparams.only_adv_train:
                preds = None
                model.set_training(training=True, bn_training=True)
                preds_adv = model.get_probs(adv_x)
            else:
                model.set_training(training=True)
                preds = model.get_probs(x)
                model.set_training(training=True, bn_training=True)
                preds_adv = model.get_probs(adv_x)
        train_fetches = self.build_train_op(preds, y, preds_adv)

        self.inputs = [self.g0_inputs]
        self.outputs = [train_fetches]

    def set_input(self, X_batch=None, Y_batch=None):
        x_pre, x, y = self.g0_inputs
        fd = {}
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

    def is_finished(self):
        """
        Single GPU trainer has no cache.
        """
        return True
