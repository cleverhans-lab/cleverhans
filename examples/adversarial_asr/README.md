# Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition

This is a Tensorflow implementation for the ICML 2019 paper ["Imperceptible, Robust and Targeted Adversarial Examples for Automatic Speech Recognition"](http://proceedings.mlr.press/v97/qin19a.html). The details of all the models implemented here can be found in the [paper](http://proceedings.mlr.press/v97/qin19a.html).

## Dependencies
*   Python 2.7
*   a TensorFlow [installation](https://www.tensorflow.org/install/) (Tensorflow 1.13 is supported for this version of Lingvo system),
*   a `C++` compiler (only g++ 4.8 is officially supported),
*   the bazel build system,
*   librosa (```pip install librosa```),
*   Cython (```pip install Cython```),
*   pyroomacoustics (```pip install pyroomacoustics```).

## Data 
Here we provide 10 audios from LibriSpeech test-clean dataset as an example to show how to run the codes. Please refer to [Lingvo](https://github.com/tensorflow/lingvo/tree/master/lingvo/tasks/asr/tools) or [Librispeech website](http://www.openslr.org/resources/12/) to download the whole test set.

In the file ```read_data.txt```, the directory of the 10 audios, the corresponding original transcription and the targeted transcription are provided in the format of [dir, original transcription, targeted transcription]. The full list of 1000 audio examples used in our experiments is provided in ```./util/read_data_ful.txt```.

You can run the script ```sh util/convert_name_format.sh``` to convert the audios in the LibriSpeech from the format ```.flac```  to ```.wav```. You need to first change the directory of the downloaded LibriSpeech dataset in the script ```./util/convert_name_format.sh```.

## Pretrained model
The pretrained model can be downloaded [here](http://cseweb.ucsd.edu/~yaq007/ckpt-00908156.data-00000-of-00001). You need to place the downloaded pretrained model into the directory ```./model/```.

## Lingvo ASR system

The automatic speech recognition (ASR) system used in this paper is [Lingvo system](https://github.com/tensorflow/lingvo). To run our codes, you need to first download the forked version [here](https://github.com/yaq007/lingvo) and make sure that you are in the "icml" branch.

```bash
git clone https://github.com/yaq007/lingvo.git
cd lingvo
git checkout icml
```
Then you need to compile the lingvo system. The easiest way to build [Lingvo system](https://github.com/tensorflow/lingvo) is to use the [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/). Here we place the folder ```lingvo/``` and ```lingvo_compiled/``` under the root directory ```~/```. If you change their locations, you need to make corresponding changes in the following commands.


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
The folder ```lingvo/``` in the directory ```lingvo_compiled/``` needs to be placed in the directory ```./adversarial_asr/```. Then this directory becomes ```./adversarial_asr/lingvo/```.

## Imperceptible Adversarial Examples

Currently, all the python scripts are tested on one GPU. You can use ```CUDA_VISIBLE_DEVICES=GPU_INDEX``` to choose which gpu to run the python scripts.

To generate imperceptible adversarial examples, run

```bash
python generate_imperceptible_adv.py
```

The adversarial examples saved with the name ended with "stage1" is the adversarial examples in [Carlini's work](https://arxiv.org/abs/1801.01944). Adversarial examples ended with the name "stage2" is our imperceptible adversarial examples using frequency masking threshold.

To test the accuracy of our imperceptible adversarial examples, simply run:

```bash
python test_imperceptible_adv.py --stage=stage2 --adv=True
```
You can set ```--stage=stage1``` to test the accuracy of Carlini's adversarial examples. If you set ```--adv=False```, then you can test the performance for clean examples with its corresponding original transcriptions.

## Robust Adversarial Examples
To generate robust adversarial examples that are simulated playing over-the-air in the simulated random rooms, we need to first generate the simulated room reverberations.
```bash
python room_simulator.py
```
Then you can run the following command to generate robust adversarial examples.
```
python generate_robust_adv.py --initial_bound=2000 --num_iter_stage1=2000
```
In the paper, we test the last 100 audios in the ```./util/read_data_full.txt``` and we set the parameter ```initial bound``` and ```num_iter_stage1``` as ```2000``` in our experiments.

Empirically, for longer audios, you might need to increase the ```initial bound``` of perturbation to generate robust adversarial examples that can successfully attack the simulated rooms. Correspndingly, you also need to increase ```num_iter_stage1``` to allow the adversarial generation to converge. You can tune the training parameters in ```generate_robust_adv.py``` to play with your data.

To test the performance of robust adversarial examples, simply run 
```
python test_robust_adv.py --stage=stage2 --adv=True
```
If you want to test the performance of clean examples played in the simulated rooms, you can set ```--adv=False```.

## Citation
If you find the code or the models implemented here are useful, please cite this paper:

```
@InProceedings{pmlr-v97-qin19a,
  title = 	 {Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition},
  author = 	 {Qin, Yao and Carlini, Nicholas and Cottrell, Garrison and Goodfellow, Ian and Raffel, Colin},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {5231--5240},
  year = 	 {2019},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  publisher = 	 {PMLR},
}
```

## Acknowledgement
This code is based on Lingvo ASR system. Thanks to the contributors of the Lingvo.

```
@article{shen2019lingvo,
  title={Lingvo: a modular and scalable framework for sequence-to-sequence modeling},
  author={Shen, Jonathan and Nguyen, Patrick and Wu, Yonghui and Chen, Zhifeng and Chen, Mia X and Jia, Ye and Kannan, Anjuli and Sainath, Tara and Cao, Yuan and Chiu, Chung-Cheng and others},
  journal={arXiv preprint arXiv:1902.08295},
  year={2019}
}
```

