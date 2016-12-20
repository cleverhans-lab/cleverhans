from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import theano
import theano.tensor as T


from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_th import th_model_train, th_model_eval, batch_eval
from cleverhans.attacks import fgsm

def main():
    """
    Test the accuracy of the MNIST cleverhans tutorial model
    :return:
    """
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

    # Define input Theano placeholder
    x_shape = (None, 1, 28, 28)
    y_shape = (None, 10)
    x = T.tensor4('x')
    y = T.matrix('y')
    
    # Define Theano model graph
    model = model_mnist()
    model.build(x_shape)
    predictions = model(x)
    print("Defined Theano model graph.")

    # Train an MNIST model
    th_model_train(x, y, predictions, model.trainable_weights, X_train, Y_train, args=args)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = th_model_eval(x, y, predictions, X_test, Y_test, args=args)
    assert float(accuracy) >= 0.98, accuracy

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    adv_x = fgsm(x, predictions, eps=0.3, back='th')
    X_test_adv, = batch_eval([x], [adv_x], [X_test], args=args)
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the MNIST model on adversarial examples
    accuracy = th_model_eval(x, y, predictions, X_test_adv, Y_test, args=args)
    assert float(accuracy) <= 0.1, accuracy
if __name__ == '__main__':
    main()
