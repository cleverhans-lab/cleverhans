import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_keras import accuracy
from cleverhans.attacks import saliency

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')


def main(argv=None):
    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'th':
        keras.backend.set_image_dim_ordering('th')
        print "INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'tf', temporarily setting to 'th'"

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print "Created TensorFlow session and set Keras backend."

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist()
    print "Loaded MNIST test data."

    # build the Keras model, and the
    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    model, f_x = model_mnist(logits=True, input_ph=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train,
              nb_epoch=FLAGS.nb_epochs,
              batch_size=FLAGS.batch_size,
              validation_split=0.15,
              verbose=1,
              shuffle=True)
    acc = accuracy(model, X_test, Y_test, batch_size=FLAGS.batch_size)
    print('Accuracy on the test set: %0.2f%%' % (100*acc))

    # Let's pick a random test sample to try our adversarial method on
    index = np.random.randint(0, X_test.shape[0])
    # Let's choose a target class that is not the correct class
    label = np.argmax(Y_test[index])
    target = np.random.choice([i for i in range(FLAGS.nb_classes) if i != label])
    print('Correct label: %i' % label)
    print('Target label: %i' % target)
    x_orig = X_test[index:(index+1)]
    x_adv = saliency(sess, x, f_x, sample=x_orig,
                     target=target, theta=1, gamma=0.1,
                     clip_min=0, clip_max=1,
                     increase=True, back='tf')


if __name__ == '__main__':
    main()
