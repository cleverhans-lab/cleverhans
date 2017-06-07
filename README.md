# cleverhans (v1.0.0)

<img src="https://github.com/openai/cleverhans/blob/master/logo.png?raw=true" alt="cleverhans logo">

[![Build Status](https://travis-ci.org/openai/cleverhans.svg?branch=master)](https://travis-ci.org/openai/cleverhans)

This repository contains the source code for `cleverhans`, a Python library to
benchmark machine learning systems' vulnerability to
[adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/).
You can learn more about such vulnerabilities on the accompanying [blog](http://cleverhans.io).

The `cleverhans` library is under continual development, always welcoming
contributions of the latest attacks and defenses.
In particular, we always welcome help towards resolving the [issues](https://github.com/openai/cleverhans/issues)
currently open.

## Setting up `cleverhans`

### Dependencies

This library uses `TensorFlow` or `Theano` to accelerate graph
computations performed by many machine learning models.
Installing these libraries is therefore a pre-requisite.
You can find instructions
[here for Tensorflow](https://www.tensorflow.org/get_started/os_setup)
and [here for Theano](http://deeplearning.net/software/theano/install.html).
For better performance, it is also recommended to install the
backend library of your choice (`TensorFlow` or `Theano`) with GPU support.

Some models used in the tutorials are also defined using `Keras`.
Note that you should **configure Keras to use the backend that matches
the one used by the tutorial**. Indeed, some tutorials use `Tensorflow`
as their backend while others use `Theano`. You
can find instructions for
setting the Keras backend [on this page](https://keras.io/backend/).

Installing `TensorFlow` or `Theano` will
take care of all other dependencies like `numpy` and `scipy`.

### Installing

Once dependencies have been taken care of, you can install `cleverhans` using
`pip` or by cloning the Github repository.

#### `pip` installation

If you are installing `cleverhans` using `pip`, run the following command:

```
pip install -e git+http://github.com/openai/cleverhans.git#egg=cleverhans
```

#### Manual installation

If you are installing `cleverhans` manually, you simply need to clone this
repository into a folder of your choice.

```
git clone https://github.com/openai/cleverhans
```

On UNIX machines, it is recommended to add your clone of this repository to the
`PYTHONPATH` variable so as to be able to import `cleverhans` from any folder.

```
export PYTHONPATH="/path/to/cleverhans":$PYTHONPATH
```

You may want to make that change permanent through your shell's profile.

## Tutorials

To help you get started with the functionalities provided by this library, the
`tutorials/' folder comes with the following tutorials:
* **MNIST with FGSM using the TensorFlow backend** ([code](tutorials/mnist_tutorial_tf.py), [tutorial](tutorials/mnist_tutorial_tf.md)): this first
tutorial covers how to train a MNIST model using TensorFlow,
craft adversarial examples using the [fast gradient sign method](https://arxiv.org/abs/1412.6572),
and make the model more robust to adversarial
examples using adversarial training.
* **MNIST with JSMA using the Tensorflow backend** ([code](tutorials/mnist_tutorial_jsma.py), [tutorial](tutorials/mnist_tutorial_jsma.md)): this second
tutorial covers how to train a MNIST model using TensorFlow and
craft adversarial examples using the [Jacobian-based saliency map approach](https://arxiv.org/abs/1511.07528).
* **MNIST with FGSM using the Theano backend** ([code](tutorials/mnist_tutorial_th.py)): this
tutorial covers how to train a MNIST model using Theano,
craft adversarial examples using the fast gradient sign
method and make the model more robust to
adversarial examples using adversarial training.
Note: this script does not have a tutorial markdown
yet, but the corresponding [tutorial](tutorials/mnist_tutorial_tf.md) in TensorFlow
will prove useful in the meanwhile.
* **MNIST using a black-box attack** ([code](tutorials/mnist_blackbox.py)):
this tutorial implements the black-box
attack described in this [paper](https://arxiv.org/abs/1602.02697).
The adversary train a substitute model: a copy that imitates the black-box
model by observing the labels that the black-box model assigns to inputs chosen
carefully by the adversary. The adversary then uses the substitute
model’s gradients to find adversarial examples that are misclassified by the
black-box model as well.

## Examples

The `examples/` folder contains additional scripts to showcase different uses
of the `cleverhans` library.

## Reporting benchmarks

When reporting benchmarks, please:
* Use a versioned release of `cleverhans`. You can find a list of released versions [here](https://github.com/openai/cleverhans/releases).
* Either use the latest version, or, if comparing to an earlier publication, use the same version as the earlier publication.
* Report which attack method was used.
* Report any configuration variables used to determine the behavior of the attack.

For example, you might report "We benchmarked the robustness of our method to
adversarial attack using v1.0.0 of `cleverhans`. On a test set modified by the
`fgsm` with `eps` of 0.3, we obtained a test set accuracy of 71.3%."

## Contributing

Contributions are welcomed! We ask that new efforts and features be coordinated
on the mailing list for `cleverhans` development: [cleverhans-dev@googlegroups.com](https://groups.google.com/forum/#!forum/cleverhans-dev).
When making contributions to `cleverhans`, we ask that you follow the
`PEP8` coding style in your pull requests.

Bug fixes can be initiated through Github pull requests.

## Citing this work

If you use `cleverhans` for academic research, you are highly encouraged
(though not required) to cite the following [paper](https://arxiv.org/abs/1610.00768):

```
@article{papernot2016cleverhans,
  title={cleverhans v1.0.0: an adversarial machine learning library},
  author={Papernot, Nicolas and Goodfellow, Ian and Sheatsley, Ryan and Feinman, Reuben and McDaniel, Patrick},
  journal={arXiv preprint arXiv:1610.00768},
  year={2016}
}
```

A new version of the technical report will be uploaded for each major
revision. GitHub contributors will be added to the author list.

## About the name

The name `cleverhans` is a reference to a presentation by Bob Sturm titled
“Clever Hans, Clever Algorithms: Are Your Machine Learnings Learning What You
Think?" and the corresponding publication, ["A Simple Method to Determine if a
Music Information Retrieval System is a
'Horse'."](http://ieeexplore.ieee.org/document/6847693/) Clever Hans was a
horse that appeared to have learned to answer arithmetic questions, but had in
fact only learned to read social cues that enabled him to give the correct
answer. In controlled settings where he could not see people's faces or receive
other feedback, he was unable to answer the same questions. The story of Clever
Hans is a metaphor for machine learning systems that may achieve very high
accuracy on a test set drawn from the same distribution as the training data,
but that do not actually understand the underlying task and perform poorly on
other inputs.

## Authors

This library is managed and maintained by Ian Goodfellow (Google Brain),
Nicolas Papernot (Pennsylvania State University), and
Ryan Sheatsley (Pennsylvania State University).

The following authors contributed (ordered according to the GitHub contributors page):
* Nicolas Papernot (Pennsylvania State University)
* Ian Goodfellow (OpenAI)
* Ryan Sheatsley (Pennsylvania State University)
* Reuben Feinman (Symantec)

## Copyright

Copyright 2017 - Google Inc., OpenAI and Pennsylvania State University.
