from model import MLPnGPU
from model import Conv2DnGPU
from model import LinearnGPU

from cleverhans_tutorials.tutorial_models import ReLU
from cleverhans_tutorials.tutorial_models import Flatten
from cleverhans_tutorials.tutorial_models import Softmax
from cleverhans_tutorials.tutorial_models import MaxPool

from resnet_tf import ResNetTF


def make_madry(nb_classes=10, input_shape=(None, 28, 28, 1), **kwargs):
    layers = [Conv2DnGPU(32, (5, 5), (1, 1), "SAME", name='conv1'),
              ReLU(name='act1'),
              MaxPool((2, 2), (2, 2), "SAME", name='mxpool1'),
              Conv2DnGPU(64, (5, 5), (1, 1), "SAME", name='conv2'),
              ReLU(name='act2'),
              MaxPool((2, 2), (2, 2), "SAME", name='mxpool2'),
              Flatten(name='flat1'),
              LinearnGPU(1024, name='fc1'),
              ReLU(name='act3'),
              LinearnGPU(nb_classes, name='fc2'),
              Softmax(name='softmax')]

    model = MLPnGPU(layers, input_shape, name='madry', **kwargs)
    return model


def make_model(model_type='madry', *args, **kwargs):
    if model_type == 'madry':
        return make_madry(*args, **kwargs)
    elif model_type == 'resnet_tf':
        return ResNetTF(*args, **kwargs)
    else:
        raise Exception('model type not defined.')
