# CleverHans (latest release: v4.0.0)

<img src="https://github.com/tensorflow/cleverhans/blob/master/assets/logo.png?raw=true" alt="cleverhans logo">


This repository contains the source code for CleverHans, a Python library to
benchmark machine learning systems' vulnerability to
[adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/).
You can learn more about such vulnerabilities on the accompanying [blog](http://cleverhans.io).

The CleverHans library is under continual development, always welcoming
[contributions](https://github.com/cleverhans-lab/cleverhans#contributing)
of the latest attacks and defenses.
In particular, we always welcome help towards resolving the [issues](https://github.com/cleverhans-lab/cleverhans/issues)
currently open.

Since v4.0.0, CleverHans supports 3 frameworks: JAX, PyTorch, and TF2. We are currently prioritizing implementing 
attacks in PyTorch, but we very much welcome contributions for all 3 frameworks. In versions v3.1.0 and prior,
CleverHans supported TF1; the code for v3.1.0 can be found under `cleverhans_v3.1.0/` or by checking
out a prior Github release.

The library focuses on providing reference implementation of attacks
against machine learning models to help with benchmarking models against
adversarial examples. 

The directory structure is as follows: 
`cleverhans/` contain attack implementations, `tutorials/` contain scripts demonstrating the features
of CleverHans, and `defenses/` contains defense implementations. Each framework has its own subdirectory
within these folders, e.g. `cleverhans/jax`.

## Setting up CleverHans

### Dependencies

This library uses [Jax](https://github.com/google/jax), [PyTorch](https://pytorch.org/) or [TensorFlow 2](https://www.tensorflow.org/) to accelerate graph
computations performed by many machine learning models.
Therefore, installing one of these libraries is a pre-requisite.

### Installation

Once dependencies have been taken care of, you can install CleverHans using
`pip` or by cloning this Github repository.

#### `pip` installation

If you are installing CleverHans using `pip`, run the following command:

```
pip install cleverhans
```

This will install the last version uploaded to
[Pypi](https://pypi.org/project/cleverhans).
If you'd instead like to install the bleeding edge version, use:

```
pip install git+https://github.com/cleverhans-lab/cleverhans.git#egg=cleverhans
```

#### Installation for development

If you want to make an editable installation of CleverHans so that you can
develop the library and contribute changes back, first fork the repository
on GitHub and then clone your fork into a directory of your choice:

```
git clone https://github.com/<your-org>/cleverhans
```

You can then install the local package in "editable" mode in order to add it to
your `PYTHONPATH`:

```
cd cleverhans
pip install -e .
```

### Currently supported setups

Although CleverHans is likely to work on many other machine configurations, we
currently test it with Python
3.6, Jax 0.2, PyTorch 1.7, and Tensorflow 2.4 on Ubuntu 18.04 LTS (Bionic Beaver).

## Getting support

If you have a request for support, please ask a question
on [StackOverflow](https://stackoverflow.com/questions/tagged/cleverhans)
rather than opening an issue in the GitHub tracker. The GitHub
issue tracker should *only* be used to report bugs or make feature requests.

## Contributing

Contributions are welcomed! To speed the code review process, we ask that:
* New efforts and features be coordinated on the [discussion board](https://github.com/cleverhans-lab/cleverhans/discussions).
* When making code contributions to CleverHans, you should follow the [`Black`](https://black.readthedocs.io/en/stable/index.html)
 coding style in your pull requests.
* We do not accept pull requests that add git submodules because of [the
  problems that arise when maintaining git
  submodules](https://medium.com/@porteneuve/mastering-git-submodules-34c65e940407).

Bug fixes can be initiated through Github pull requests.

## Tutorials: `tutorials` directory

To help you get started with the functionalities provided by this library, the
`tutorials/` folder comes with the following tutorials:
* **MNIST with FGSM and PGD** ([jax](tutorials/jax/mnist_tutorial.py), [tf2](tutorials/tf2/mnist_tutorial.py):
this tutorial covers how to train an MNIST model and craft adversarial examples using the
 [fast gradient sign method](https://arxiv.org/abs/1412.6572) and 
 [projected gradient descent](https://arxiv.org/abs/1706.06083).
* **CIFAR10 with FGSM and PGD** ([pytorch](tutorials/torch/cifar10_tutorial.py), [tf2](tutorials/tf2/cifar10_tutorial.py)):
this tutorial covers how to train a CIFAR10 model and 
craft adversarial examples using the [fast gradient sign method](https://arxiv.org/abs/1412.6572) and
 [projected gradient descent](https://arxiv.org/abs/1706.06083).

NOTE: the tutorials are maintained carefully, in the sense that we use
continuous integration to make sure they continue working. They are not
considered part of the API and they can change at any time without warning.
You should not write 3rd party code that imports the tutorials and expect
that the interface will not break. Only the main library is subject to
our six month interface deprecation warning rule.

NOTE: please start a thread on the [discussion board](https://github.com/cleverhans-lab/cleverhans/discussions) before writing a new
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

Since we recently discontinued support for TF1, the `examples/` folder is currently 
empty, but you are welcome to submit your uses via a pull request :)

Old examples for CleverHans v3.1.0 and prior can be found under `cleverhans_v3.1.0/examples/`.

## Reporting benchmarks

When reporting benchmarks, please:
* Use a versioned release of CleverHans. You can find a list of released versions [here](https://github.com/cleverhans-lab/cleverhans/releases).
* Either use the latest version, or, if comparing to an earlier publication, use the same version as the earlier publication.
* Report which attack method was used.
* Report any configuration variables used to determine the behavior of the attack.

For example, you might report "We benchmarked the robustness of our method to
adversarial attack using v4.0.0 of CleverHans. On a test set modified by the
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

This library is collectively maintained by the [CleverHans Lab](https://cleverhans-lab.github.io/) 
at the University of Toronto. The current point of contact is Jonas Guan. 
It was previously maintained by Ian Goodfellow and Nicolas Papernot.


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
* Anshuman Suri (University of Virginia)
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
* Erh-Chung Chen (National Tsing Hua University)
* Joel Frank (Ruhr-University Bochum)

## Copyright

Copyright 2021 - Google Inc., OpenAI, Pennsylvania State University, University of Toronto.
