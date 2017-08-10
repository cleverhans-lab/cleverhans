"""
This example shows the transferability of adversarial samples
both within models and across models.
This property of adversarial samples is heavily used in both
attacks and defenses.
https://arxiv.org/pdf/1412.6572.pdf
has more discussions on the same.
This code uses pure Tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.version import LooseVersion
import numpy as np
import math

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod

from tutorials.mnist_tutorial_pure_tf import MLP, Linear, Layer, Conv2D, ReLU 
from tutorials.mnist_tutorial_pure_tf import Softmax, Flatten

FLAGS = flags.FLAGS

def make_basic_cnn(nb_filters=64, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):

    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def make_basic_fc(nb_classes=10,
                  input_shape=(None, 28, 28, 1)):
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]
  
    model = MLP(layers, input_shape)
    return model
  

def make_basic_cnn_2(nb_filters=64, nb_classes=10,
                     input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Flatten(),
              Linear(32),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def confusion(sess, x, y, preds, X_test, Y_test, batch_size):
    """
    Function that prints confusion matrix
    """
    if LooseVersion(tf.__version__) >= LooseVersion('1.0.0'):
        predictions = tf.argmax(preds, axis=-1)
        labels = tf.argmax(y, axis=-1)
    else:
        predictions = tf.argmax(preds, axis=tf.rank(preds) - 1)
        labels = tf.argmax(y, axis=tf.rank(y) - 1)

    predictionList = np.array(list())
    labelList = np.array(list())
    with sess.as_default():

        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        for batch in range(nb_batches):

            start = batch * batch_size
            end = min(len(X_test), start + batch_size)
            cur_batch_size = end - start

            batch_predictions, batch_labels = sess.run([predictions, labels], 
                                              feed_dict={x: X_test[start:end],
                                              y: Y_test[start:end]})

            predictionList = np.append(predictionList, batch_predictions)
            labelList = np.append(labelList, batch_labels)

    matrix = tf.confusion_matrix(predictions=predictionList, labels=labelList)
    with sess.as_default():
        print(matrix.eval())
    return

def mnist_transferability_example(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.1):
    """
    MNIST Transferability example in pure tensorflow
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    returns None
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model.fprop(x)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                args=train_params)
    
    # Print initial confusion matrix, without adversarial samples
    print("Confusion Matrix for legitimate examples for CNN with two layers:")
    confusion(sess, x, y, preds, X_test, Y_test, batch_size=batch_size)
    print("The above confusion matrix shows that after training, very", 
          "few samples are misclassified.")

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': 0.3}
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.fprop(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    
    # Print confusion Matrix
    print("Confusion Matrix for adversarial samples generated and evaluated",
          "on the same two layer CNN:")
    confusion(sess, x, y, preds_adv, X_test, Y_test, batch_size=batch_size)
    print("On the other hand, when adversarial inputs are used, most",
           "predictions are wrong.")

    # Train a second model to show transferability across models
    print("A second model with the same structure is indepedently trained")
    model_2 = make_basic_cnn()
    preds_2 = model_2(x)

    def evaluate_2():
        # Evaluate accuracy on legitimate test samples
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)

    # Train second model
    model_train(sess, x, y, preds_2, X_train, Y_train, evaluate=evaluate_2,
                args=train_params, init_all=False)

    # Evaluate the accuracy of the second model using adversarial
    # samples from the first
    preds_adv_2 = model_2(adv_x)
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv_2, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)

    # Print the confusion matrix
    print("Confusion Matrix for adversarial examples generated and evaluated",
          "on two different two layer CNN with same architectures, trained",
          "independantly:")
    confusion(sess, x, y, preds_adv_2, X_test, Y_test, batch_size=batch_size)
    print("While the accuracy is not as low as below, it is still far below",
          "acceptable")
    
    print("Two more models are trained, this time with different structures")
    # Train a third model to show transferability across models
    model_3 = make_basic_cnn()
    preds_3 = model_3(x)

    def evaluate_3():
        # Evaluate accuracy on legitimate test samples
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_3, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)

    # Train third model
    model_train(sess, x, y, preds_3, X_train, Y_train, evaluate=evaluate_3,
                args=train_params, init_all=False)

    # Evaluate the accuracy of the third model using adversarial
    # samples from the first
    preds_adv_3 = model_3(adv_x)
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv_3, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)

    # Print the confusion matrix
    print("Confusion Matrix for adversarial examples generated from a two",
          "layer CNN and evaluated on a two hidden layer fully connected",
          "network:")
    confusion(sess, x, y, preds_adv_3, X_test, Y_test, batch_size=batch_size)

    # Train a fourth model to show transferability across models
    model_4 = make_basic_cnn_2()
    preds_4 = model_4(x)

    def evaluate_4():
        # Evaluate accuracy on legitimate test samples
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_4, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)

    # Train fourth model
    model_train(sess, x, y, preds_4, X_train, Y_train, evaluate=evaluate_4,
                args=train_params, init_all=False)

    # Evaluate the accuracy of the fourth model using adversarial
    # samples from the first
    preds_adv_4 = model_4(adv_x)
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv_4, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)

    # Print the confusion matrix
    print("Confusion Matrix for adversarial samples generated from a two",
          "layer CNN and evaluated on a two layer CNN with an additional",
          "fully connected hidden layer:")
    confusion(sess, x, y, preds_adv_4, X_test, Y_test, batch_size=batch_size)
    print("The transfer of adversarial samples across models is clearly",
          "seen above.")

    return


def main(argv=None):
    mnist_transferability_example(nb_epochs=FLAGS.nb_epochs,
                                  batch_size=FLAGS.batch_size,
                                  learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')

    app.run()
