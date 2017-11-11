import logging

from evaluator import create_adv_by_name
from trainer import TrainManager

from runner import RunnerSingleGPU


class TrainerSingleGPU(TrainManager):
    def __init__(self, *args, **kwargs):
        super(TrainerSingleGPU, self).__init__(*args, **kwargs)
        self.runner = RunnerSingleGPU(self.inputs, self.outputs,
                                      sess=self.sess)

    def create_train_graph(self):
        # The evaluation graph should be initialized after the train graph is
        # fully initialized, otherwise, some of the variables will be created
        # untrainable.
        assert self.evaluate is None, ("""Evaluation graph should be initialzed
                                       after the train graph""")
        self.model.set_device('/gpu:0')
        hparams = self.hparams
        model = self.model
        x = self.g0_inputs['x']
        y = self.g0_inputs['y']
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

    def sync_params(self, forced=False):
        """
        Nothing to sync on single GPU.
        """
        return True
