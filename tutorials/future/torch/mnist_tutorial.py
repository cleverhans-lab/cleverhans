import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import cleverhans.future.torch.attacks as attacks

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

######
# load MNIST
######
train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ],
        )
test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ],
        )
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=train_transforms,
    download=True
    )
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=test_transforms,
    download=True
    )
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
    )
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2
    )

######
# build and train network
######
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

net = LeNet5(1, 2)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

def train(n_epoch):
  print('training starts')
  net.train()
  for _ in range(n_epoch):
    for __, (x, y) in enumerate(train_loader):
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      loss = loss_fn(net(x), y)
      loss.backward()
      optimizer.step()

      print('epoch: {}/{}, batch: {}/{}, batch loss({}): {:.3f}'.format(
          _+1, n_epoch, __+1, len(train_loader), loss_fn.__class__.__name__, loss.item()))

def adv_train(n_epoch):
  """
  Adversarial training from the Madry Lab challenge.
  """
  print('adversarial training starts')
  net.train()
  pgd = attacks.ProjectedGradientDescent(net)
  for _ in range(n_epoch):
    for __, (x, y) in enumerate(train_loader):
      x, y = x.to(device), y.to(device)
      x_adv_pgd = pgd.generate(
        x,
        y=y,
        eps=.3,
        eps_iter=.01,
        nb_iter=40,
        ord=np.inf
        )
      optimizer.zero_grad()
      loss = loss_fn(net(x_adv_pgd), y)
      loss.backward()
      optimizer.step()

      print('epoch: {}/{}, batch: {}/{}, batch loss({}): {:.3f}'.format(
          _+1, n_epoch, __+1, len(train_loader), loss_fn.__class__.__name__, loss.item()))

def test():
  net.eval()
  with torch.no_grad():
    total = 0
    total_correct = 0

    for x, y in test_loader:
      x, y = x.to(device), y.to(device)
      _, y_pred = net(x).max(1)
      total += y.size(0)
      total_correct += y_pred.eq(y).sum().item()

    print('test acc (%): {:.3f}'.format(total_correct/total * 100))

if torch.cuda.device_count() > 1:
  net = torch.nn.DataParallel(net)
  net.to(device)

# train(10)
adv_train(200)
test()

######
# generate adversarial examples
######
fgm = attacks.FastGradientMethod(net)
pgd = attacks.ProjectedGradientDescent(net)

total = 0
total_correct_fgm = 0
total_correct_pgd = 0

for __, (x, y) in enumerate(test_loader):
  x, y = x.to(device), y.to(device)
  x_adv_fgm = fgm.generate(
      x,
      y=y,
      eps=.3,
      ord=np.inf
      )
  _, y_pred_fgm = net(x_adv_fgm).max(1)
  total_correct_fgm += y_pred_fgm.eq(y).sum().item()

  x_adv_pgd = pgd.generate(
      x,
      y=y,
      eps=.3,
      eps_iter=.01,
      nb_iter=40,
      ord=np.inf
      )
  _, y_pred_pgd = net(x_adv_pgd).max(1)
  total += y.size(0)
  total_correct_pgd += y_pred_pgd.eq(y).sum().item()
  print('batch {}/{}'.format(__+1, len(test_loader)))

print('fgm test acc (%): {:.3f}'.format(total_correct_fgm/total * 100))
print('pgd test acc (%): {:.3f}'.format(total_correct_pgd/total * 100))
