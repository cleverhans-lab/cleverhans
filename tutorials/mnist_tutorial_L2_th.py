from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import theano
import theano.tensor as T

import cPickle as pickle

import numpy as np

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_th import th_model_train, th_model_eval, batch_eval
from cleverhans.attacks_th import carlini_L2


def main():
    """
    MNIST cleverhans tutorial
    :return:
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', default=1000, type=int, help='Size of training batches')
    parser.add_argument('--train_dir', '-d', default='/tmp', help='Directory storing the saved model.')
    parser.add_argument('--filename', '-f',  default='mnist.ckpt', help='Filename to save model under.')
    parser.add_argument('--nb_epochs', '-e', default=6, type=int, help='Number of epochs to train model')
    parser.add_argument('--nb_iters', '-i', default=10000, type=int, help='Number of iterations for crafting adversarial examples')
    parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='Learning rate for training')
    parser.add_argument('--eps', default=0.01, type=float, help='Epsilon for Carlini L2 Attack')
    parser.add_argument('--kappa', default=0.01, type=float, help='Kappa for Carlini L2 Attack')
    parser.add_argument('--c', default=20, type=float)
    parser.add_argument('--load', default=None, type=str, help='Model path to load')
    parser.add_argument('--dump', default=None, type=str, help='Model path to dump')
    args = parser.parse_args()

    np.random.seed(126)

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
    
    if args.load:
        model = pickle.load(open(args.load, "rb"))
        predictions = model(x)
    else:
        # Define Theano model graph
        model = model_mnist()
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

        if args.dump:
            pickle.dump(model, open(args.dump, "wb"))

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    #for i in range(10):
    carlini_L2(x, predictions, X_test, Y_test, eps=args.eps, kappa=args.kappa, c=args.c, nb_iters=args.nb_iters, batch_size=args.batch_size)
    

if __name__ == '__main__':
    main()
