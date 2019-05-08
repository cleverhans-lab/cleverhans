import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

from cleverhans.future.tf2.attacks import projected_gradient_descent, fast_gradient_method

FLAGS = flags.FLAGS

class CNN(Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.conv2 = Conv2D(64, 3, activation='relu')
    self.conv3 = Conv2D(128, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
    
   
def load_cifar10():
  """Load CIFAR10 training and test data."""

  def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  dataset, info = tfds.load('cifar10', data_dir='gs://tfds-data/datasets', with_info=True,
                            as_supervised=True)
  mnist_train, mnist_test = dataset['train'], dataset['test']
  mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(128)
  mnist_test = mnist_test.map(convert_types).batch(128)
  return EasyDict(train=mnist_train, test=mnist_test)
  
  
def main(_):
  # Load training and test data
  data = load_mnist()
  #Load CNN Model
  model = CNN()
  loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.optimizers.Adam()

  # Metrics to track the different accuracies.
  train_loss = tf.metrics.Mean(name='train_loss')
  test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
  test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
  
  @tf.function
  def train_step(x, y):
    with tf.GradientTape() as tape:
      predictions = model(x)
      loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', 8, 'Number of epochs.')
  flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')
  flags.DEFINE_bool('adv_train', False, 'Use adversarial training (on PGD adversarial examples).')
  app.run(main)
  
