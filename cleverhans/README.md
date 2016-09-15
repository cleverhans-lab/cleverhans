# cleverhans (v0.1)

This repository contains the source code for `cleverhans`, a Python library to
benchmark adversarial examples in machine learning.

## Setting up `cleverhans`

This library uses `TensorFlow` to accelerate graph computations performed by
many machine learning models. Some models are also defined using `Keras`.
Installing these libraries with GPU support is recommended for performance.
Note that you should **configure Keras to use the TensorFlow backend**, as
explained [on this page](https://keras.io/backend/).

Installing `TensorFlow` and `Keras` should take care of all other dependencies
like `numpy` and `scipy`.

## Tutorials

This library comes with the following tutorials:
* MNIST ([code](mnist_tutorial.py), [tutorial](mnist_tutorial.md)): this first
tutorial covers how to train a MNIST model using TensorFlow,
craft adversarial examples, and make the model more robust to adversarial
examples using adversarial training.
* more to come soon...

## Authors

The following authors contributed to this library (by alphabetical order):
* Ian Goodfellow (OpenAI)
* Nicolas Papernot (Pennsylvania State University)
* Ryan Sheatsley (Pennsylvania State University)

## Copyright

Copyright 2016 - Pennsylvania State University and OpenAI.