from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import array
import gzip
import os
from os import path
import struct
from six.moves.urllib.request import urlretrieve

import numpy as np
import torch

_DATA = "/tmp/data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urlretrieve(url, out_file)
        print("downloaded {} to {}".format(url, _DATA))


def mnist_raw(root=_DATA):
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(root, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(root, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(root, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(root, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


class MNISTDataset(torch.utils.data.Dataset):
    """MNIST Dataset."""

    def __init__(self, root=_DATA, train=True, transform=None):
        train_images, train_labels, test_images, test_labels = mnist_raw(root=root)

        if train:
            self.images = train_images
            self.labels = torch.from_numpy(train_labels).long()
        else:
            self.images = test_images
            self.labels = torch.from_numpy(test_labels).long()

        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.images)
