from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent

FLAGS = flags.FLAGS


class LeNet5(torch.nn.Module):
  """
  LeNet-5
  (N, in_channels, 32, 32) in, (N, 120) out
  """

  def __init__(self, in_channels, padding):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=padding)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


def ld_mnist():
  train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=train_transforms, download=True)
  test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=test_transforms, download=True)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
  return EasyDict(train=train_loader, test=test_loader)


def main(_):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Load training and test data
  data = ld_mnist()

  # Instantiate model, loss, and optimizer for training
  net = LeNet5(1, 2)
  loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

  # This will be called later to report accuracy on clean and adversarial examples
  def eval():
    net.eval()
    total = 0
    total_correct = 0
    total_correct_fgm = 0
    total_correct_pgd = 0
    for x, y in data.test:
      x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
      x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40)
      _, y_pred = net(x).max(1)  # model prediction on clean examples
      _, y_pred_fgm = net(x_fgm).max(1)  # model prediction on adversarial examples
      _, y_pred_pgd = net(x_pgd).max(1)  # model prediction on adversarial examples
      total += y.size(0)
      total_correct += y_pred.eq(y).sum().item()
      total_correct_fgm += y_pred_fgm.eq(y).sum().item()
      total_correct_pgd += y_pred_pgd.eq(y).sum().item()
    print('test acc on clean examples (%): {:.3f}'.format(total_correct / total * 100))
    print('test acc on FGM adversarial examples (%): {:.3f}'.format(total_correct_fgm / total * 100))
    print('test acc on PGD adversarial examples (%): {:.3f}'.format(total_correct_pgd / total * 100))

  # Train vanilla model
  net.train()
  for epoch in range(FLAGS.nb_epochs):
    for __, (x, y) in enumerate(data.train):
      optimizer.zero_grad()
      loss = loss_fn(net(x), y)
      loss.backward()
      optimizer.step()

    print('epoch: {}/{}, last batch loss: {:.3f}'.format(epoch, FLAGS.nb_epochs, loss.item()))

  # Evaluate on clean and adversarial data
  eval()

  # Train model with adversarial training
  net.train()
  for epoch in range(FLAGS.nb_epochs):
    for __, (x, y) in enumerate(data.train):
      adv_x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40)
      optimizer.zero_grad()
      loss = loss_fn(net(adv_x), y)
      loss.backward()
      optimizer.step()

    print('epoch: {}/{}, last batch loss: {:.3f}'.format(epoch, FLAGS.nb_epochs, loss.item()))

  # Evaluate on clean and adversarial data
  eval()

if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs.')
  flags.DEFINE_float('eps', 0.3, 'Total epsilon for FGM and PGD attacks.')


  app.run(main)
