from model import MLPnGPU
from model import Conv2DnGPU
from model import LinearnGPU
from model import MaxPool

from cleverhans_tutorials.tutorial_models import ReLU
from cleverhans_tutorials.tutorial_models import Flatten
from cleverhans_tutorials.tutorial_models import Softmax

from resnet_tf import ResNetTF


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

    model = MLPnGPU(layers, input_shape, **kwargs)
    return model


def make_model(model_type='madry', *args, **kwargs):
    if model_type == 'madry':
        return make_madry_ngpu(*args, **kwargs)
    elif model_type == 'resnet_tf':
        return ResNetTF(*args, **kwargs)
    else:
        raise Exception('model type not defined.')
