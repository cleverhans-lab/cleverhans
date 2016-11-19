import keras

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import tf_model_train, tf_model_eval, batch_eval
from cleverhans.attacks import jsma

import os

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'mnist.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 28, 'Input row dimension')
flags.DEFINE_integer('img_cols', 28, 'Input column dimension')
flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_float('per_samples', 0.01, 'Percentage of test set to attack')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')


import numpy as np
def main(argv=None):
    """
    MNIST cleverhans tutorial for the Jacobian-based saliency map approach (JSMA)
    :return:
    """
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

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
    y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

    # Define TF model graph
    model = model_mnist()
    predictions = model(x)
    print "Defined TensorFlow model graph."

    # Train an MNIST model, name="inputTensor")
    tf_model_train(sess, x, y, predictions, X_train, Y_train)

    #saver = tf.train.Saver()
    #save_path = saver.save(sess, os.path.join(FLAGS.train_dir, 'model.ckpt'))
    #saver.restore(sess, "/tmp/model.ckpt")

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
    assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    # Craft adversarial examples for nb_classes from per_samples using the Jacobian-based saliency map approach 
    num_samples = int(len(X_test)*FLAGS.per_samples)
    results = np.zeros((FLAGS.nb_classes, num_samples), dtype='i,f')
    print 'Crafting ' + str(num_samples) + ' * ' + str(FLAGS.nb_classes) + ' adversarial examples'
    for sample in xrange(num_samples):
        for target in xrange(FLAGS.nb_classes):
            print '------------------------------------------\nCreating adversarial example for target class ' + str(target)
            adv_x, result, percentage_perterb = jsma(sess, x, predictions, X_test[sample:(sample + 1)], target, 
                    theta=1, gamma=0.1, increase=True, back='tf', clip_min=0, clip_max=1)
            results[target, sample] = (result, percentage_perterb)

    num_suceeded = 0
    percentage_perterbed = 0.0
    for y in xrange(num_samples):
        for x in xrange(FLAGS.nb_classes):
            if results[x, y][0] == 1:
                num_suceeded = num_suceeded + 1
            percentage_perterbed = percentage_perterbed + results[x, y][1]

    percentage_perterbed = percentage_perterbed / float(num_samples * FLAGS.nb_classes)
    print('Avg. percent of successful misclassifcations {0} \n avg. percent perterbed {1}'\
            .format(float(num_suceeded)/float(FLAGS.nb_classes - 1), percentage_perterbed))

if __name__ == '__main__':
    app.run()
