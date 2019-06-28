from absl import app, flags

import datasets
import itertools
import time
import jax.numpy as np
import numpy.random as npr
from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, logsoftmax

from cleverhans.future.jax.attacks import fast_gradient_method, projected_gradient_descent

FLAGS = flags.FLAGS

def main(_):
  rng = random.PRNGKey(0)

  # Load MNIST dataset
  train_images, train_labels, test_images, test_labels = datasets.mnist()

  batch_size = 128
  batch_shape = (-1, 28, 28, 1)
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  train_images = np.reshape(train_images, batch_shape)
  test_images = np.reshape(test_images, batch_shape)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  # Model, loss, and accuracy functions
  init_random_params, predict = stax.serial(
          stax.Conv(32, (8, 8), strides=(2, 2), padding='SAME'),
          stax.Relu,
          stax.Conv(128, (6, 6), strides=(2, 2), padding='VALID'),
          stax.Relu,
          stax.Conv(128, (5, 5), strides=(1, 1), padding='VALID'),
          stax.Flatten,
          stax.Dense(128),
          stax.Relu,
          stax.Dense(10))

  def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -np.mean(logsoftmax(preds) * targets)

  def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

  # Instantiate an optimizer
  opt_init, opt_update, get_params = optimizers.adam(0.001)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  # Initialize model
  _, init_params = init_random_params(rng, batch_shape)
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  # Training loop
  print("\nStarting training...")
  for epoch in range(FLAGS.nb_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time

    # Evaluate model on clean data
    params = get_params(opt_state)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))

    # Evaluate model on adversarial data
    model_fn = lambda images: predict(params, images)
    test_images_fgm = fast_gradient_method(model_fn, test_images, FLAGS.eps, np.inf)
    test_images_pgd = projected_gradient_descent(model_fn, test_images, FLAGS.eps, 0.01, 40, np.inf)
    test_acc_fgm = accuracy(params, (test_images_fgm, test_labels))
    test_acc_pgd = accuracy(params, (test_images_pgd, test_labels))

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy: {}".format(train_acc))
    print("Test set accuracy on clean examples: {}".format(test_acc))
    print("Test set accuracy on FGM adversarial examples: {}".format(test_acc_fgm))
    print("Test set accuracy on PGD adversarial examples: {}".format(test_acc_pgd))

if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', 8, 'Number of epochs.')
  flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')

  app.run(main)
