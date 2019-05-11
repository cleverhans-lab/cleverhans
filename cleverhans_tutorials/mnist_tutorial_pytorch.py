"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numpy as np
import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001


warnings.warn("convert_pytorch_model_to_tf is deprecated, switch to"
              + " dedicated PyTorch support provided by CleverHans v4.")


class PytorchMnistModel(nn.Module):
  """ Basic MNIST model from github
  https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb
  """

  def __init__(self):
    super(PytorchMnistModel, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    self.fc1 = nn.Linear(64 * 7 * 7, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64 * 7 * 7)  # reshape Variable
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=-1)


def mnist_tutorial(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   train_end=-1, test_end=-1, learning_rate=LEARNING_RATE):
  """
  MNIST cleverhans tutorial
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :return: an AccuracyReport object
  """
  # Train a pytorch MNIST model
  torch_model = PytorchMnistModel()
  if torch.cuda.is_available():
    torch_model = torch_model.cuda()
  report = AccuracyReport()

  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('data', train=True, download=True,
                     transform=transforms.ToTensor()),
      batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
      batch_size=batch_size)

  # Truncate the datasets so that our test run more quickly
  train_loader.dataset.train_data = train_loader.dataset.train_data[
      :train_end]
  test_loader.dataset.test_data = test_loader.dataset.test_data[:test_end]

  # Train our model
  optimizer = optim.Adam(torch_model.parameters(), lr=learning_rate)
  train_loss = []

  total = 0
  correct = 0
  step = 0
  for _epoch in range(nb_epochs):
    for xs, ys in train_loader:
      xs, ys = Variable(xs), Variable(ys)
      if torch.cuda.is_available():
        xs, ys = xs.cuda(), ys.cuda()
      optimizer.zero_grad()
      preds = torch_model(xs)
      loss = F.nll_loss(preds, ys)
      loss.backward()  # calc gradients
      train_loss.append(loss.data.item())
      optimizer.step()  # update gradients

      preds_np = preds.cpu().detach().numpy()
      correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
      total += train_loader.batch_size
      step += 1
      if total % 1000 == 0:
        acc = float(correct) / total
        print('[%s] Training accuracy: %.2f%%' % (step, acc * 100))
        total = 0
        correct = 0

  # Evaluate on clean data
  total = 0
  correct = 0
  for xs, ys in test_loader:
    xs, ys = Variable(xs), Variable(ys)
    if torch.cuda.is_available():
      xs, ys = xs.cuda(), ys.cuda()

    preds = torch_model(xs)
    preds_np = preds.cpu().detach().numpy()

    correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
    total += len(xs)

  acc = float(correct) / total
  report.clean_train_clean_eval = acc
  print('[%s] Clean accuracy: %.2f%%' % (step, acc * 100))

  # We use tf for evaluation on adversarial data
  sess = tf.Session()
  x_op = tf.placeholder(tf.float32, shape=(None, 1, 28, 28,))

  # Convert pytorch model to a tf_model and wrap it in cleverhans
  tf_model_fn = convert_pytorch_model_to_tf(torch_model)
  cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

  # Create an FGSM attack
  fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
  fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}
  adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
  adv_preds_op = tf_model_fn(adv_x_op)

  # Run an evaluation of our model against fgsm
  total = 0
  correct = 0
  for xs, ys in test_loader:
    adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
    correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
    total += test_loader.batch_size

  acc = float(correct) / total
  print('Adv accuracy: {:.3f}'.format(acc * 100))
  report.clean_train_adv_eval = acc
  return report


def main(_=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)

  mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                 batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  tf.app.run()
