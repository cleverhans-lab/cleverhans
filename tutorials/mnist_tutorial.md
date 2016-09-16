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
Also, add the path to the repository cloneto your 
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


