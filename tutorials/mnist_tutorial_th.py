from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import theano
import theano.tensor as T

import keras
from keras import backend

from cleverhans.utils import cnn_model
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_th import th_model_train, th_model_eval, batch_eval
from cleverhans.attacks_th import fgsm


def main():
    """
    MNIST cleverhans tutorial
    :return:
    """

    if not hasattr(backend, "theano"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the Theano backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', default=128, help='Size of training batches')
    parser.add_argument('--train_dir', '-d', default='/tmp', help='Directory storing the saved model.')
    parser.add_argument('--filename', '-f',  default='mnist.ckpt', help='Filename to save model under.')
    parser.add_argument('--nb_epochs', '-e', default=6, type=int, help='Number of epochs to train model')
    parser.add_argument('--learning_rate', '-lr', default=0.5, type=float, help='Learning rate for training')
    args = parser.parse_args()

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    print("Loaded MNIST test data.")

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input Theano placeholder
    x_shape = (None, 1, 28, 28)
    y_shape = (None, 10)
    x = T.tensor4('x')
    y = T.matrix('y')

    # Define Theano model graph
    model = cnn_model()
    model.build(x_shape)
    predictions = model(x)
    print("Defined Theano model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = th_model_eval(x, y, predictions, X_test, Y_test, args=args)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: ' + str(accuracy))
        pass

    # Train an MNIST model
    th_model_train(x, y, predictions, model.trainable_weights, X_train, Y_train, evaluate=evaluate, args=args)


    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3)
    X_test_adv, = batch_eval([x], [adv_x], [X_test], args=args)
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = th_model_eval(x, y, predictions, X_test_adv, Y_test, args=args)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

    print("Repeating the process, using adversarial training")
    # Redefine Theano model graph
    model_2 = cnn_model()
    model_2.build(x_shape)
    predictions_2 = model_2(x)
    adv_x_2 = fgsm(x, predictions_2, eps=0.3)
    predictions_2_adv = model_2(adv_x_2)


    def evaluate_2():
        # Evaluate the accuracy of the adversarialy trained MNIST model on
        # legitimate test examples
        accuracy = th_model_eval(x, y, predictions_2, X_test, Y_test, args=args)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

        # Evaluate the accuracy of the adversarially trained MNIST model on
        # adversarial examples
        accuracy_adv = th_model_eval(x, y, predictions_2_adv, X_test, Y_test, args=args)
        print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

    # Perform adversarial training
    th_model_train(x, y, predictions_2, model_2.trainable_weights, X_train, Y_train, predictions_adv=predictions_2_adv,
            evaluate=evaluate_2, args=args)



if __name__ == '__main__':
    main()
