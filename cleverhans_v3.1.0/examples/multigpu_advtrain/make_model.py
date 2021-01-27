# pylint: disable=missing-docstring
from model import Conv2D, ReLU, Flatten, Linear, Softmax, MLP
from model import MLPnGPU
from model import Conv2DnGPU
from model import LinearnGPU
from model import MaxPool

from resnet_tf import ResNetTF


def make_basic_cnn(nb_filters=64, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):
  layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
            ReLU(),
            Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
            ReLU(),
            Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
            ReLU(),
            Flatten(),
            Linear(nb_classes),
            Softmax()]

  model = MLP(nb_classes, layers, input_shape)
  return model


def make_basic_ngpu(nb_classes=10, input_shape=(None, 28, 28, 1), **kwargs):
  """
  Create a multi-GPU model similar to the basic cnn in the tutorials.
  """
  model = make_basic_cnn()
  layers = model.layers

  model = MLPnGPU(nb_classes, layers, input_shape)
  return model


def make_madry_ngpu(nb_classes=10, input_shape=(None, 28, 28, 1), **kwargs):
  """
  Create a multi-GPU model similar to Madry et al. (arXiv:1706.06083).
  """
  layers = [Conv2DnGPU(32, (5, 5), (1, 1), "SAME"),
            ReLU(),
            MaxPool((2, 2), (2, 2), "SAME"),
            Conv2DnGPU(64, (5, 5), (1, 1), "SAME"),
            ReLU(),
            MaxPool((2, 2), (2, 2), "SAME"),
            Flatten(),
            LinearnGPU(1024),
            ReLU(),
            LinearnGPU(nb_classes),
            Softmax()]

  model = MLPnGPU(nb_classes, layers, input_shape)
  return model


def make_model(model_type='madry', **kwargs):
  if model_type == 'basic':
    return make_basic_ngpu(**kwargs)
  elif model_type == 'madry':
    return make_madry_ngpu(**kwargs)
  elif model_type == 'resnet_tf':
    return ResNetTF(**kwargs)
  else:
    raise Exception('model type not defined.')
