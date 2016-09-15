from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def data_mnist():
    """
    Preprocess MNIST dataset
    :return:
    """
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, FLAGS.img_rows, FLAGS.img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, FLAGS.img_rows, FLAGS.img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, FLAGS.nb_classes)
    Y_test = np_utils.to_categorical(y_test, FLAGS.nb_classes)
    return X_train, Y_train, X_test, Y_test


def model_mnist(tf_placeholder=None):
    """
    Defines MNIST model using Keras sequential model
    :param tf_placeholder:
    :return:
    """
    model = Sequential()

    model.add(Convolution2D(FLAGS.nb_filters, 5, 5,
                            border_mode='valid',
                            input_shape=(1, FLAGS.img_rows, FLAGS.img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(FLAGS.nb_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(FLAGS.nb_pool, FLAGS.nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if tf_placeholder is None:
        return model
    else:
        return model(tf_placeholder)
