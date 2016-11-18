import numpy as np


def accuracy(model, X, y, batch_size=128):
    """
    Test the model accuracy on a holdout set.
    :param model: A Keras model.
    :param X: input features
    :param y: input labels
    :return:
    """
    class_preds = model.predict_classes(X, batch_size=batch_size)
    class_true = np.where(y)[1]
    nb_correct = np.where(class_preds == class_true)[0].shape[0]
    return float(nb_correct)/class_preds.shape[0]