from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import theano.tensor as T
import keras

from cleverhans.utils import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_th import th_model_train, th_model_eval


def main():
    keras.backend.set_image_dim_ordering('th')

    # We can't use argparse in a test because it reads the arguments to nosetests
    # e.g., nosetests -v passes the -v to the test
    args = {
            "batch_size": 128,
            "nb_epochs": 2,
            "learning_rate": .5
            }

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    X_train = X_train[:10000]
    Y_train = Y_train[:10000]
    X_test = X_test[:2000]
    Y_test = Y_test[:2000]

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input Theano placeholder
    x_shape = (None, 1, 28, 28)
    x = T.tensor4('x')
    y = T.matrix('y')

    # Define Theano model graph
    model = cnn_model()
    model.build(x_shape)
    predictions = model(x)
    print("Defined Theano model graph.")

    # Train an MNIST model
    th_model_train(x, y, predictions, model.trainable_weights,
                   X_train, Y_train, args=args)

    accuracy = th_model_eval(x, y, predictions, X_test, Y_test, args=args)

    assert accuracy > 0.8, accuracy

def test():
    main()

if __name__ == '__main__':
    main()
