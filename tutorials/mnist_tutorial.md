# MNIST tutorial

This tutorial explains how to use `cleverhans` together 
with a TensorFlow model to craft adversarial examples, 
as well as make the model more robust to adversarial 
examples. We assume basic knowledge of TensorFlow. 

## Setup

First, make sure that you have [TensorFlow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#download-and-setup) 
and [Keras](https://keras.io/#installation) installed on
your machine and then clone the `cleverhans` 
[repository](https://github.com/openai/cleverhans).
Also, add the path of the repository clone to your 
`PYTHONPATH` environment variable. 
```
export PYTHONPATH="/path/to/cleverhans":$PYTHONPATH
```
This allows our tutorial script to import the library 
simply with `import cleverhans`. 

The tutorial's [complete script](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial.py) 
is provided in the `tutorial` folder of the 
`cleverhans` repository. 

## Defining the model with TensorFlow and Keras

In this tutorial, we use Keras to define the model
and TensorFlow to train it. The model is a Keras 
[`Sequential` model](https://keras.io/models/sequential/): 
it is made up of multiple convolutional and ReLU layers. 
You can find the model definition in the 
[`utils_mnist` cleverhans module](https://github.com/openai/cleverhans/blob/master/cleverhans/utils_mnist.py).

TODO(insert code snippet here)

## Training the model with TensorFlow

