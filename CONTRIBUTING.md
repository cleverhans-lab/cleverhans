# Contributing to CleverHans

First off, thank you for considering contributing to CleverHans.
Following these guidelines helps to communicate that you respect
the time of the researchers and developers managing and developing this open
source project. In return, they should reciprocate that respect in
addressing your issue, assessing changes, and helping you finalize
your pull requests.

Adding new features, improving documentation, bug triaging, or
writing tutorials are all
examples of helpful contributions.
Furthermore, if you are publishing a new attack or defense,
we strongly encourage you to add it to CleverHans so that others
may evaluate it fairly in their own work.

To speed the code review process, we ask that:
* New efforts and features be coordinated on the [discussion board](https://github.com/cleverhans-lab/cleverhans/discussions).
* When making code contributions to CleverHans, you should follow the
[`Black`](https://black.readthedocs.io/en/stable/index.html) coding style in your pull requests.
* We do not accept pull requests that add git submodules because of [the
  problems that arise when maintaining git
  submodules](https://medium.com/@porteneuve/mastering-git-submodules-34c65e940407)

Bug fixes can be initiated through Github pull requests.

## Development setup

Please follow the usual 
[git forking workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) 
when contributing.

### Setting up Cleverhans on your machine

Then create a new Conda or Virtualenv environment. 

Conda:
```
$ conda create --name cleverhans python=3.6
$ conda activate cleverhans
```

Virtualenv:
```
$ python3 -m venv /path/to/new/virtual/environment
$ cd /path/to/new/virtual/environment
$ source ./bin/activate
```

Then, after `cd`-ing into the `cleverhans` directory, install the 
Cleverhans library and all corresponding requirements into your 
newly created environment.

```
$ pip install -e "."
$ pip install -r requirements/requirements.txt
$ pip install -r requirements/requirements-pytorch.txt
$ pip install -r requirements/requirements-jax.txt
$ pip install -r requirements/requirements-tf2.txt
$ pip install -r requirements/requirements-dev.txt
```

Optionally also install GPU dependencies for JAX (PyTorch and 
TF2 already come with GPU as part of their default package):
```
$ pip install -r requirements/requirements-gpu.txt
```

### Add git pre-commit hooks

Install our pre-commit hooks that ensure that your code is always formatted
via `black` before committing.

```
$ pre-commit install
```

Note that we do have code style checks in place for every submitted 
PR and will reject PRs that do not meet these checks. By installing the 
pre-commit hooks, this will be taken care of automatically