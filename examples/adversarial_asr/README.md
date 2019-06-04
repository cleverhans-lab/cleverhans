# Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition

### Introduction
This is a Tensorflow implementation for the ICML 2019 paper "Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition". The details of all the models implemented here can be found in the [paper](http://proceedings.mlr.press/v97/qin19a.html).

## Dependencies

*   a TensorFlow [installation](https://www.tensorflow.org/install/) (Tensorflow 1.13.1 is required for this version of Lingvo system),
*   a `C++` compiler (only g++ 4.8 is officially supported),
*   the bazel build system,
*   librosa (pip install librosa).


## Quick Start

The automatic speech recognition (ASR) system used in this paper is [Lingvo system](https://github.com/tensorflow/lingvo). You can download the forked version [here](https://github.com/yaq007/lingvo) and make sure that you are in the "icml" branch.

```bash
git clone https://github.com/yaq007/lingvo.git
cd lingvo
git checkout icml
```
Then we need to compile the lingvo system. The easiest way to build [Lingvo system](https://github.com/tensorflow/lingvo) is to use the docker.

```bash
cd ..
mkdir lingvo_compiled

export LINGVO_DEVICE="gpu"
sudo docker build --no-cache --tag tensorflow:lingvo $(test "$LINGVO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04") - < lingvo/docker/dev.dockerfile

export LINGVO_DIR=$HOME/lingvo
sudo docker run --rm $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo -v ~/lingvo_compiled:/tmp/lingvo_compiled -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo bash

# In docker
bazel build -c opt --config=cuda //lingvo/tools:create_asr_features
bazel build -c opt --config=cuda //lingvo:trainer
cp -rfL bazel-bin/lingvo/trainer.runfiles/__main__/lingvo /tmp/lingvo_compiled

# Outside of docker
sudo chown -R $USER ~/lingvo_compiled
export PYTHONPATH=$PYTHONPATH:~/lingvo_compiled
```
