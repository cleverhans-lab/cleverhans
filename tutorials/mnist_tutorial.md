# MNIST tutorial

This tutorial explains how to use `cleverhans` together 
with a TensorFlow model to craft adversarial examples, 
as well as make the model more robust to adversarial 
examples. We assume basic knowledge of TensorFlow. 

## Setup

First, make sure that you cloned the `cleverhans` 
[repository](https://github.com/openai/cleverhans) and
that you added it to your `PYTHONPATH` environment
variable. This allows our tutorial script to import 
the library simply with `import cleverhans`. 

The tutorial's [complete script](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial.py) 
is provided in the `tutorial` folder of the 
`cleverhans` repository. 
