from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import keras
from keras import backend as K
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers.core import Lambda
from keras.engine import Merge


from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils import model_loss
from cleverhans.attacks import fgsm



def main():
    """
    MNIST cleverhans tutorial
    :return:
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', default=128, help='Size of training batches')
    parser.add_argument('--train_dir', '-d', default='/tmp', help='Directory storing the saved model.')
    parser.add_argument('--filename', '-f',  default='mnist.ckpt', help='Filename to save model under.')
    parser.add_argument('--nb_epochs', '-e', default=6, type=int, help='Number of epochs to train model')
    parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='Learning rate for training')
    args = parser.parse_args()
    
    if keras.backend.backend() == 'tensorflow':
        import tensorflow as tf
        # Set TF random seed to improve reproducibility
        tf.set_random_seed(1234)
    
        # Image dimensions ordering should follow the Theano convention
        if keras.backend.image_dim_ordering() != 'th':
            keras.backend.set_image_dim_ordering('th')
            print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'")
    
        # Create TF session and set as Keras backend session
        sess = tf.Session()
        keras.backend.set_session(sess)
        print("Created TensorFlow session and set Keras backend.")

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    print("Loaded MNIST test data.")

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input Keras placeholder
    x = Input(shape=(1, 28, 28))
    y = Input(shape=(10,))

    # Define Keras model graph
    model = model_mnist(name='classifier')
    print("Defined Keras model graph.")

    # Train an MNIST model
    print('compiling...')
    model.compile(optimizer=Adadelta(lr=args.learning_rate,
                    rho=0.95, epsilon=1e-08),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    print('done!')
    model.fit(X_train,
              Y_train,
              nb_epoch=args.nb_epochs,
              batch_size=args.batch_size,
              validation_data=(X_test, Y_test),
              verbose=1)
    print('trained!')
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    
    predictions = model(x)
    adv_x = fgsm(x, predictions, eps=0.3)
    predictions_adv = model(adv_x)
    model_adv = Model(input=x, output=[predictions, predictions_adv])
    model_adv.compile(optimizer=Adadelta(lr=args.learning_rate,
                    rho=0.95, epsilon=1e-08),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    metrics = model_adv.evaluate(X_test, [Y_test] * 2)
    print()
    print('Test accuracy on MNIST examples: %.4f' % metrics[-2])
    print('Test accuracy on adversarial examples: %.4f' % metrics[-1])
    print()
    print("Repeating the process, using adversarial training")
    
    # Redefine Keras model graph
    x_2 = Input(shape=(1, 28, 28))
    
    model_2_classifier = model_mnist(name='classifier')
    predictions_2 = model_2_classifier(x_2)
    adv_x_2 = fgsm(x_2, predictions_2, eps=0.3)
    predictions_2_adv = model_2_classifier(adv_x_2)
    model_2 = Model(input=x_2, output=[predictions_2, predictions_2_adv])
    
    model_2.compile(optimizer=Adadelta(lr=args.learning_rate,
                    rho=0.95, epsilon=1e-08),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model_2.fit(X_train,
                [Y_train] * 2,
                nb_epoch=args.nb_epochs,
                batch_size=args.batch_size,
                validation_data=(X_test, [Y_test] * 2))

    metrics = model_2.evaluate(X_test, [Y_test] * 2)
    print()
    print('Test accuracy on MNIST examples: %.4f' % metrics[-2])
    print('Test accuracy on adversarial examples: %.4f' % metrics[-1])

if __name__ == '__main__':
    main()
