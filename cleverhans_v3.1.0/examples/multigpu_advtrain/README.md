# MultiGPU Adversarial Training

This directory includes scripts to train a model using a multi-GPU 
implementation of ["Madry et al.  (2017)"](https://arxiv.org/abs/1706.06083) 
attack.

## MNIST Adversarial Training

Train a model using a single GPU on MNIST.

```python
CUDA_VISIBLE_DEVICES=0 python run_multigpu.py --attack_type_train FGSM --attack_type_test FGSM --adv_train
```

Train using `MadryEtal_y` attack. The default number of steps on MNIST is set to 40 for this attack.

```python
CUDA_VISIBLE_DEVICES=0 python run_multigpu.py --attack_type_train MadryEtAl_y --attack_type_test FGSM,MadryEtAl_y --adv_train --save_dir runs/mnist_madry_advtrain --nb_epochs 200 --model_type madry
```

Here is the output after 200 iterations.

```text
Clean accuracy, subsampled train (train_accuracy_subsampled): 1.0000
Clean accuracy, natural test (test_accuracy_natural): 0.9870
Test accuracy on adversarial examples (test_accuracy_FGSM): 0.9633
Test accuracy on adversarial examples (test_accuracy_MadryEtAl_y): 0.9483
```

The same training can be done on 2 GPUs by setting `--ngpu 2` and 
`--attack_type_train MadryEtAl_y_multigpu`. Set the number iterations for the 
attack at training time via the flag `--attack_nb_iter_train`.

## CIFAR10 Adversarial Training

This command trains a ResNet32 model on CIFAR10 using multi-GPU adversarial 
training.

```python
CUDA_VISIBLE_DEVICES=0,1 python run_multigpu.py --attack_type_train MadryEtAl_y_multigpu --attack_type_test FGSM,MadryEtAl_y --ngpu 2 --save_dir runs/cifar10_advtrain --nb_epochs 200 --model_type resnet_tf --attack_nb_iter_train 7 --adv_train --dataset cifar10 --eval_iters 5
```

Here is an example of the output of the command above at epoch 200.

```text
Clean accuracy, subsampled train (train_accuracy_subsampled): 0.9560
Clean accuracy, natural test (test_accuracy_natural): 0.8989
Test accuracy on adversarial examples (test_accuracy_FGSM): 0.7986
Test accuracy on adversarial examples (test_accuracy_MadryEtAl_y): 0.7803
```

## Testing

Run tests using the following commands:

```python
CUDA_VISIBLE_DEVICES=0,1 python test_run_multigpu.py
CUDA_VISIBLE_DEVICES=0,1 python test_runner.py
CUDA_VISIBLE_DEVICES=0,1 python test_attack_multigpu.py
```
