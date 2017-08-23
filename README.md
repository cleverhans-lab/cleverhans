# CleverHans (v2.0.0)

<img src="https://github.com/tensorflow/cleverhans/blob/master/assets/logo.png?raw=true" alt="cleverhans logo">

[![Build Status](https://travis-ci.org/tensorflow/cleverhans.svg?branch=master)](https://travis-ci.org/tensorflow/cleverhans)

This repository contains the source code for CleverHans, a Python library to
benchmark machine learning systems' vulnerability to
[adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/).
You can learn more about such vulnerabilities on the accompanying [blog](http://cleverhans.io).

The CleverHans library is under continual development, always welcoming
[contributions](https://github.com/tensorflow/cleverhans#contributing)
of the latest attacks and defenses.
In particular, we always welcome help towards resolving the [issues](https://github.com/tensorflow/cleverhans/issues)
currently open.

## Setting up CleverHans

### Dependencies

This library uses [TensorFlow](https://www.tensorflow.org/) to accelerate graph
computations performed by many machine learning models.
Installing TensorFlow is therefore a pre-requisite.

You can find instructions
[here](https://www.tensorflow.org/install/).
For better performance, it is also recommended to install TensorFlow
with GPU support (detailed instructions on how to do this are available
in the TensorFlow installation documentation).

Installing TensorFlow will
take care of all other dependencies like `numpy` and `scipy`.

### Installation

Once dependencies have been taken care of, you can install CleverHans using
`pip` or by cloning this Github repository.

#### `pip` installation

If you are installing CleverHans using `pip`, run the following command:

```
pip install -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans
```

#### Manual installation

If you are installing CleverHans manually, you need to install TensorFlow
first. Then, run the following command to clone the CleverHans repository
into a folder of your choice:

```
git clone https://github.com/tensorflow/cleverhans
```

On UNIX machines, it is recommended to add your clone of this repository to the
`PYTHONPATH` variable so as to be able to import `cleverhans` from any folder.

```
export PYTHONPATH="/path/to/cleverhans":$PYTHONPATH
```

You may want to make that change permanent through your shell's profile.

### Currently supported setups

Although CleverHans is likely to work on many other machine configurations, we
currently [test it](https://travis-ci.org/tensorflow/cleverhans) with Python
{2.7, 3.5} and TensorFlow {1.0, 1.1} on Ubuntu 14.04.5 LTS (Trusty Tahr).

## Tutorials

To help you get started with the functionalities provided by this library, the
`cleverhans_tutorials/' folder comes with the following tutorials:
* **MNIST with FGSM** ([code](cleverhans_tutorials/mnist_tutorial_tf.py)): this
tutorial covers how to train a MNIST model using TensorFlow,
craft adversarial examples using the [fast gradient sign method](https://arxiv.org/abs/1412.6572),
and make the model more robust to adversarial examples using adversarial training.
* **MNIST with FGSM using Keras** ([code](cleverhans_tutorials/mnist_tutorial_keras_tf.py)): this
tutorial covers how to define a MNIST model with Keras and train it using TensorFlow,
craft adversarial examples using the [fast gradient sign method](https://arxiv.org/abs/1412.6572),
and make the model more robust to adversarial
examples using adversarial training.
* **MNIST with JSMA** ([code](cleverhans_tutorials/mnist_tutorial_jsma.py)): this second
tutorial covers how to define a MNIST model with Keras and train it using TensorFlow and
craft adversarial examples using the [Jacobian-based saliency map approach](https://arxiv.org/abs/1511.07528).
* **MNIST using a black-box attack** ([code](cleverhans_tutorials/mnist_blackbox.py)):
this tutorial implements the black-box
attack described in this [paper](https://arxiv.org/abs/1602.02697).
The adversary train a substitute model: a copy that imitates the black-box
model by observing the labels that the black-box model assigns to inputs chosen
carefully by the adversary. The adversary then uses the substitute
model’s gradients to find adversarial examples that are misclassified by the
black-box model as well.

Some models used in the tutorials are defined using [Keras](https://keras.io),
which should be installed before running these tutorials.
Installation instructions for Keras can be found
[here](https://keras.io/#installation).
Note that you should configure Keras to use the TensorFlow backend. You
can find instructions for
setting the Keras backend [on this page](https://keras.io/backend/).

## Examples

The `examples/` folder contains additional scripts to showcase different uses
of the CleverHans library or get you started competing in different adversarial
example contests.

## Reporting benchmarks

When reporting benchmarks, please:
* Use a versioned release of CleverHans. You can find a list of released versions [here](https://github.com/tensorflow/cleverhans/releases).
* Either use the latest version, or, if comparing to an earlier publication, use the same version as the earlier publication.
* Report which attack method was used.
* Report any configuration variables used to determine the behavior of the attack.

For example, you might report "We benchmarked the robustness of our method to
adversarial attack using v2.0.0 of CleverHans. On a test set modified by the
`FastGradientMethod` with a max-norm `eps` of 0.3, we obtained a test set accuracy of 71.3%."

## Contributing

Contributions are welcomed! To speed the code review process, we ask that:
* New efforts and features be coordinated
on the mailing list for CleverHans development: [cleverhans-dev@googlegroups.com](https://groups.google.com/forum/#!forum/cleverhans-dev).
* When making code contributions to CleverHans, you follow the
`PEP8` coding style in your pull requests.
* When making your first pull request, you [sign the Google CLA](https://cla.developers.google.com/clas)

Bug fixes can be initiated through Github pull requests.

## Citing this work

If you use CleverHans for academic research, you are highly encouraged
(though not required) to cite the following [paper](https://arxiv.org/abs/1610.00768):

```
@article{papernot2016cleverhans,
  title={cleverhans v1.0.0: an adversarial machine learning library},
  author={Papernot, Nicolas and Goodfellow, Ian and Sheatsley, Ryan and Feinman, Reuben and McDaniel, Patrick},
  journal={arXiv preprint arXiv:1610.00768},
  year={2016}
}
```
There is not yet an ArXiv tech report for v2.0.0 but one will be prepared soon.

## About the name

The name CleverHans is a reference to a presentation by Bob Sturm titled
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

The following authors contributed 100 lines or more (ordered according to the GitHub contributors page):
* Nicolas Papernot (Pennsylvania State University, Google Brain intern)
* Nicholas Carlini (UC Berkeley)
* Ian Goodfellow (Google Brain)
* Reuben Feinman (Symantec)
* Fartash Faghri (University of Toronto, Google Brain intern)
* Alexander Matyasko (Nanyang Technological University)
* Karen Hambardzumyan (YerevaNN)
* Yi-Lin Juang (NTUEE)
* Alexey Kurakin (Google Brain)
* Ryan Sheatsley (Pennsylvania State University)
* Abhibhav Garg (IIT Delhi)
* Yen-Chen Lin (National Tsing Hua University)
* Paul Hendricks

## Copyright

Copyright 2017 - Google Inc., OpenAI and Pennsylvania State University.
