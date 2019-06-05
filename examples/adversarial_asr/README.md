# Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition

### Introduction
This is a Tensorflow implementation for the ICML 2019 paper "Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition". The details of all the models implemented here can be found in the [paper](http://proceedings.mlr.press/v97/qin19a.html).

## Dependencies

*   a TensorFlow [installation](https://www.tensorflow.org/install/) (Tensorflow 1.13 is supported for this version of Lingvo system),
*   a `C++` compiler (only g++ 4.8 is officially supported),
*   the bazel build system,
*   librosa (pip install librosa).

## Data and pretrained model
Here we provide 10 audios from LibriSpeech test-clean dataset as an example to show how to run the codes. The directory of the audios, the corresponding original transcription and the targeted transcription are provided in the file ```read_data.txt```. 

## Quick Start
### Lingvo ASR system

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
bazel build -c opt --config=cuda //lingvo:trainer
cp -rfL bazel-bin/lingvo/trainer.runfiles/__main__/lingvo /tmp/lingvo_compiled

# Outside of docker
sudo chown -R $USER ~/lingvo_compiled
export PYTHONPATH=$PYTHONPATH:~/lingvo_compiled
```

### Imperceptible Adversarial Examples
To generate imperceptible adversarial examples, you need to first place all the files in the lingvo_compiled/ into this directory adversarial_adv/ and then run

```bash
python generate_imperceptible_adv.py
```

The adversarial examples saved with the name ended with "stage1" is the adversarial examples in [Carlini's work](https://arxiv.org/abs/1801.01944). Adversarial examples ended with the name "stage2" is our imperceptible adversarial examples using frequency masking threshold.

To test the accuracy of the imperceptible adversarial examples, simply run:

```bash
python test_imperceptible_adv.py
```
