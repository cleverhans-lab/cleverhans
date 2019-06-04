# Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition

### Introduction
This is a Tensorflow implementation for the ICML 2019 paper "Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition". The details of all the models implemented here can be found in our [paper](http://proceedings.mlr.press/v97/qin19a.html).

## Dependencies

*   a TensorFlow [installation](https://www.tensorflow.org/install/) (Tensorflow v.12 is required for this version of Lingvo system),
*   a `C++` compiler (only g++ 4.8 is officially supported),
*   the bazel build system,
*   librosa (pip install librosa).


## Quick Start

The ASR system that is used in this paper is [Lingvo system](https://github.com/tensorflow/lingvo). 

```bash
cd lingvo
git checkout icml
cd ..
mkdir lingvo_compiled

LINGVO_DEVICES="gpu"
sudo docker build --no-cache --tag tensorflow:lingvo $(test "$LINGVO_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04") - < lingvo/docker/dev.dockerfile

export LINGVO_DIR=$HOME/lingvo
sudo docker run --rm $(test "$LINGVO_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${LINGVO_DIR}:/tmp/lingvo -v ~/lingvo_compiled:/tmp/lingvo_compiled -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name lingvo tensorflow:lingvo bash


```
