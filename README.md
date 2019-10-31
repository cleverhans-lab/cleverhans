# CleverHans (latest release: v3.0.1)

<img src="https://github.com/tensorflow/cleverhans/blob/master/assets/logo.png?raw=true" alt="cleverhans logo">

[![Build Status](https://travis-ci.org/tensorflow/cleverhans.svg?branch=master)](https://travis-ci.org/tensorflow/cleverhans)
[![Documentation Status](https://readthedocs.org/projects/cleverhans/badge/?version=latest)](https://cleverhans.readthedocs.io/en/latest/?badge=latest)

This repository contains the source code for CleverHans, a Python library to
benchmark machine learning systems' vulnerability to
[adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/).
You can learn more about such vulnerabilities on the accompanying [blog](http://cleverhans.io).

The CleverHans library is under continual development, always welcoming
[contributions](https://github.com/tensorflow/cleverhans#contributing)
of the latest attacks and defenses.
In particular, we always welcome help towards resolving the [issues](https://github.com/tensorflow/cleverhans/issues)
currently open.

## Major updates coming to CleverHans

CleverHans will soon support 3 frameworks: JAX, PyTorch, and TF2.  The package
itself will focus on its initial principle: reference implementation of attacks
against machine learning models to help with benchmarking models against
adversarial examples. This repository will also contain two folders:
`tutorials/` for scripts demonstrating the features of CleverHans and
`defenses/` for scripts that contain authoritative implementations of defenses
in one of the 3 supported frameworks. The structure of the future repository
will look like this:

```
cleverhans/
  jax/
    attacks/
      ...
    tests/
      ...
  tf2/
    attacks/
      ...
    tests/
      ...
  torch/
    attacks/
      ...
    tests/
      ...
defenses/
  jax/
    ...
  tf2/
    ...
  torch/
    ...
tutorials/
  jax/
    ...
  tf2/
    ...
  torch/
    ...
```

In the meanwhile, all of these folders can be found in the correspond `future/`
subdirectory (e.g., `cleverhans/future/jax/attacks`, `cleverhans/future/jax/tests` or `defenses/future/jax/`).

A public milestone has been created to track the changes that are to be
implemented before the library version is incremented to v4. 

## Setting up CleverHans

### Dependencies

This library uses [TensorFlow](https://www.tensorflow.org/) to accelerate graph
computations performed by many machine learning models.
Therefore, installing TensorFlow is a pre-requisite.

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

If you are installing CleverHans using `pip`, run the following command
after installing TensorFlow:

```
pip install cleverhans
```

This will install the last version uploaded to
[Pypi](https://pypi.org/project/cleverhans).
If you'd instead like to install the bleeding edge version, use:

```
pip install git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans
```

#### Installation for development

If you want to make an editable installation of CleverHans so that you can
develop the library and contribute changes back, first fork the repository
on GitHub and then clone your fork into a directory of your choice:

```
git clone https://github.com/tensorflow/cleverhans
```

You can then install the local package in "editable" mode in order to add it to
your `PYTHONPATH`:

```
cd cleverhans
pip install -e .
```

### Currently supported setups

Although CleverHans is likely to work on many other machine configurations, we
currently [test it](https://travis-ci.org/tensorflow/cleverhans) it with Python
3.5 and TensorFlow {1.8, 1.12} on Ubuntu 14.04.5 LTS (Trusty Tahr).
Support for Python 2.7 is deprecated.
CleverHans 3.0.1 supports Python 2.7 and the master branch is likely to
continue to work in Python 2.7 for some time, but we no longer run the tests
in Python 2.7 and we do not plan to fix bugs affecting only Python 2.7 after
2019-07-04.
Support for TensorFlow prior to 1.12 is deprecated.
Backwards compatibility wrappers for these versions may be removed after 2019-07-07,
and we will not fix bugs for those versions after that date.
Support for TensorFlow 1.7 and earlier is already deprecated: we do not fix
bugs for those versions and any remaining wrapper code for those versions
may be removed without further notice.

## Getting support

If you have a request for support, please ask a question
on [StackOverflow](https://stackoverflow.com/questions/tagged/cleverhans)
rather than opening an issue in the GitHub tracker. The GitHub
issue tracker should *only* be used to report bugs or make feature requests.

## Contributing

Contributions are welcomed! To speed the code review process, we ask that:
* New efforts and features be coordinated
on the mailing list for CleverHans development: [cleverhans-dev@googlegroups.com](https://groups.google.com/forum/#!forum/cleverhans-dev).
* When making code contributions to CleverHans, you follow the
`PEP8 with two spaces` coding style (the same as the one used
by TensorFlow) in your pull requests.
In most cases this can be done by running `autopep8 -i --indent-size 2 <file>`
on the files you have edited.
You can check your code by running `nosestests cleverhans/devtools/tests/test_format.py` or check an individual file by running `pylint <file>` from inside the cleverhans repository root directory.
* When making your first pull request, you [sign the Google CLA](https://cla.developers.google.com/clas)
* We do not accept pull requests that add git submodules because of [the
  problems that arise when maintaining git
  submodules](https://medium.com/@porteneuve/mastering-git-submodules-34c65e940407)

Bug fixes can be initiated through Github pull requests.

## Scripts: `scripts` directory

The `scripts` directory contains command line utilities.
In many cases you can use these to run CleverHans functionality on your
saved models without needing to write any of your own Python code.

You may want to set your `.bashrc` / `.bash_profile` file to add the
CleverHans `scripts` directory to your `PATH` environment variable
so that these scripts will be conveniently executable from any directory.

## Tutorials: `cleverhans_tutorials` directory

To help you get started with the functionalities provided by this library, the
`cleverhans_tutorials/` folder comes with the following tutorials:
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

NOTE: the tutorials are maintained carefully, in the sense that we use
continuous integration to make sure they continue working. They are not
considered part of the API and they can change at any time without warning.
You should not write 3rd party code that imports the tutorials and expect
that the interface will not break. Only the main library is subject to
our six month interface deprecation warning rule.

NOTE: please write to cleverhans-dev@googlegroups.com before writing a new
tutorial. Because each new tutorial involves a large amount of duplicated
code relative to the existing tutorials, and because every line of code
requires ongoing testing and maintenance indefinitely, we generally prefer
not to add new tutorials. Each tutorial should showcase an extremely different
way of using the library. Just calling a different attack, model, or dataset
is not enough to justify maintaining a parallel tutorial.

## Examples : `examples` directory

The `examples/` folder contains additional scripts to showcase different uses
of the CleverHans library or get you started competing in different adversarial
example contests. We do not offer nearly as much ongoing maintenance or support
for this directory as the rest of the library, and if code in here gets broken
we may just delete it without warning.

## List of attacks

You can find a full list attacks along with their function signatures at [cleverhans.readthedocs.io](http://cleverhans.readthedocs.io/)

## Reporting benchmarks

When reporting benchmarks, please:
* Use a versioned release of CleverHans. You can find a list of released versions [here](https://github.com/tensorflow/cleverhans/releases).
* Either use the latest version, or, if comparing to an earlier publication, use the same version as the earlier publication.
* Report which attack method was used.
* Report any configuration variables used to determine the behavior of the attack.

For example, you might report "We benchmarked the robustness of our method to
adversarial attack using v3.0.1 of CleverHans. On a test set modified by the
`FastGradientMethod` with a max-norm `eps` of 0.3, we obtained a test set accuracy of 71.3%."

## Citing this work

If you use CleverHans for academic research, you are highly encouraged
(though not required) to cite the following [paper](https://arxiv.org/abs/1610.00768):

```
@article{papernot2018cleverhans,
  title={Technical Report on the CleverHans v2.1.0 Adversarial Examples Library},
  author={Nicolas Papernot and Fartash Faghri and Nicholas Carlini and
  Ian Goodfellow and Reuben Feinman and Alexey Kurakin and Cihang Xie and
  Yash Sharma and Tom Brown and Aurko Roy and Alexander Matyasko and
  Vahid Behzadan and Karen Hambardzumyan and Zhishuai Zhang and
  Yi-Lin Juang and Zhi Li and Ryan Sheatsley and Abhibhav Garg and
  Jonathan Uesato and Willi Gierke and Yinpeng Dong and David Berthelot and
  Paul Hendricks and Jonas Rauber and Rujun Long},
  journal={arXiv preprint arXiv:1610.00768},
  year={2018}
}
```

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

This library is managed and maintained by Ian Goodfellow (Google Brain) and
Nicolas Papernot (Google Brain).

The following authors contributed 100 lines or more (ordered according to the GitHub contributors page):
* Ian Goodfellow (Google Brain)
* Nicolas Papernot (Google Brain)
* Nicholas Carlini (Google Brain)
* Fartash Faghri (University of Toronto)
* Tzu-Wei Sung (National Taiwan University)
* Alexey Kurakin (Google Brain)
* Reuben Feinman (New York University)
* Shiyu Duan (University of Florida)
* Phani Krishna (Video Analytics Lab)
* David Berthelot (Google Brain)
* Tom Brown (Google Brain)
* Cihang Xie (Johns Hopkins)
* Yash Sharma (The Cooper Union)
* Aashish Kumar (HARMAN X)
* Aurko Roy (Google Brain)
* Alexander Matyasko (Nanyang Technological University)
* Anshuman Suri (Microsoft)
* Yen-Chen Lin (MIT)
* Vahid Behzadan (Kansas State)
* Jonathan Uesato (DeepMind)
* Florian Tramèr (Stanford University)
* Haojie Yuan (University of Science & Technology of China)
* Zhishuai Zhang (Johns Hopkins)
* Karen Hambardzumyan (YerevaNN)
* Jianbo Chen (UC Berkeley)
* Catherine Olsson (Google Brain)
* Aidan Gomez (University of Oxford)
* Zhi Li (University of Toronto)
* Yi-Lin Juang (NTUEE)
* Pratyush Sahay (formerly HARMAN X)
* Abhibhav Garg (IIT Delhi)
* Aditi Raghunathan (Stanford University)
* Yang Song (Stanford University)
* Riccardo Volpi (Italian Institute of Technology)
* Angus Galloway (University of Guelph)
* Yinpeng Dong (Tsinghua University)
* Willi Gierke (Hasso Plattner Institute)
* Bruno López
* Jonas Rauber (IMPRS)
* Paul Hendricks (NVIDIA)
* Ryan Sheatsley (Pennsylvania State University)
* Rujun Long (0101.AI)
* Bogdan Kulynych (EPFL)
* Erfan Noury (UMBC)
* Robert Wagner (Case Western Reserve University)

## Copyright

Copyright 2019 - Google Inc., OpenAI and Pennsylvania State University.
