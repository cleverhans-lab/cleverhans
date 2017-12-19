from cleverhans_tutorials.tutorial_models import Conv2D
from cleverhans_tutorials.tutorial_models import ReLU
from cleverhans_tutorials.tutorial_models import Flatten
from cleverhans_tutorials.tutorial_models import Linear
from cleverhans_tutorials.tutorial_models import Softmax
from cleverhans_tutorials.tutorial_models import MLP


def make_imagenet_cnn(input_shape=(None, 224, 224, 3)):
    layers = [Conv2D(96, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(256, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(384, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(384, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(256, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Linear(4096),
              ReLU(),
              Linear(4096),
              ReLU(),
              Linear(1000),
              Softmax()]

    model = MLP(layers, input_shape)
    return model
