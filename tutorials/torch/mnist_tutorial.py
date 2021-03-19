from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import MNISTDataset

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


FLAGS = flags.FLAGS


class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1
        )  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.conv2 = nn.Conv2d(
            64, 128, 6, 2
        )  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.conv3 = nn.Conv2d(
            128, 128, 5, 1
        )  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.fc1 = nn.Linear(
            128 * 4 * 4, 128
        )  # (batch_size, 128, 4, 4) --> (batch_size, 2048)
        self.fc2 = nn.Linear(128, 10)  # (batch_size, 128) --> (batch_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class PyNet(nn.Module):
    """CNN architecture. This is the same MNIST model from pytorch/examples/mnist repository"""

    def __init__(self, in_channels=1):
        super(PyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def ld_mnist():
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    # Load MNIST dataset
    train_dataset = MNISTDataset(root="/tmp/data", transform=train_transforms)
    test_dataset = MNISTDataset(
        root="/tmp/data", train=False, transform=test_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


def main(_):
    # Load training and test data
    data = ld_mnist()

    # Instantiate model, loss, and optimizer for training
    if FLAGS.model == "cnn":
        net = CNN(in_channels=1)

    elif FLAGS.model == "pynet":
        net = PyNet(in_channels=1)
    else:
        raise NotImplementedError

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Train vanilla model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )

    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        x_fgm = fast_gradient_method(net, x, FLAGS.eps, np.inf)
        x_pgd = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, np.inf)
        _, y_pred = net(x).max(1)  # model prediction on clean examples
        _, y_pred_fgm = net(x_fgm).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd = net(x_pgd).max(
            1
        )  # model prediction on PGD adversarial examples
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report.correct_fgm / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report.correct_pgd / report.nb_test * 100.0
        )
    )


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 8, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )
    flags.DEFINE_enum("model", "cnn", ["cnn", "pynet"], "Choose model type.")

    app.run(main)
