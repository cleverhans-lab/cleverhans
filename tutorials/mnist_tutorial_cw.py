from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

import os
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils import other_classes, cnn_model
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax, tf_model_load

FLAGS = flags.FLAGS


def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                        test_end=10000, viz_enabled=True, nb_epochs=6,
                        batch_size=128, nb_classes=10, source_samples=10,
                        learning_rate=0.1):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Disable Keras learning phase since we will be serving through tensorflow
    keras.layers.core.K.set_learning_phase(0)

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Image dimensions ordering should follow the TensorFlow convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' "
              "to 'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = cnn_model()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': os.path.join(*os.path.split(FLAGS.model_path)[:-1]),
        'filename': os.path.split(FLAGS.model_path)[-1]
        
    }

    # check if we've trained before, and if we have, use that pre-trained model
    if os.path.exists(FLAGS.model_path+".meta"):
        tf_model_load(sess, FLAGS.model_path)
    else:
        model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                    save=os.path.exists("models"))

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1)
          + ' adversarial examples')
    print("This could take some time ...")

    # Initialize our array for grid visualization
    grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')


    # by default, we have softmax after a CNN, remove it here
    model.layers.pop()
    last = model.layers[-1]
    last.outbound_nodes = []
    model.outputs = [last.output]
    model.built = False

    # Check if we are using a GPU, and generate higher quality attacks if we are
    try:
        with tf.device('/gpu:0'):
            sess.run(tf.constant(1))
        gpu = True
    except:
        gpu = False
        
    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, back='tf', sess=sess)
    cw_params = {'binary_search_steps':1, 'max_iterations':1000 if gpu else 100,
                 'learning_rate':0.1, 'targeted':True, 'batch_size':100,
                 'initial_const': 10}

    onehot = np.zeros((10,10))
    onehot[np.arange(10),np.arange(10)] = 1

    idxs = [np.where(np.argmax(Y_test,axis=1)==i)[0][0] for i in range(10)]
    adv_inputs = np.array([[x]*10 for x in X_test[idxs]], dtype=np.float32).reshape((100,28,28,1))
    adv_ys = np.array([onehot]*10, dtype=np.float32).reshape((100,10))

    adv = sess.run(cw.generate(tf.constant(adv_inputs), tf.constant(adv_ys), **cw_params))

    adv_accuracy = model_eval(sess, x, y, preds, adv, adv_ys, args={'batch_size': 100})

    for j in range(10):
        for i in range(10):
            grid_viz_data[i,j] = adv[i*10+j]
    
    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1.-adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv-adv_inputs)**2,axis=(1,2,3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
        import matplotlib.pyplot as plt
        _ = grid_visual(grid_viz_data)

    return report


def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        nb_classes=FLAGS.nb_classes,
                        source_samples=FLAGS.source_samples,
                        learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models","mnist"), 'Path to save or load the model file')

    app.run()
