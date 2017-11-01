This directory includes scripts to train a model using iterative attacks.

The following command trains a model using a single GPU on MNIST:

```python
CUDA_VISIBLE_DEVICES=7 python run_multigpu.py --save_dir runs/X
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- run_multigpu.py --attack_type_train FGSM --attack_type_test FGSM --adv_train
```

This command trains a model using 2 GPUs on MNIST.

```python
CUDA_VISIBLE_DEVICES=0,1 ipython --pdb -- run_multigpu.py --attack_type_train MadryEtAl_y_multigpu --attack_type_test FGSM --adv_train --ngpu 2 --attack_nb_iter_train 1
```

This command trains a model using multiple GPUs on CIFAR10.

```python
python run_multigpu.py --save_dir runs/X --ngpu 8 --fast_tests --attack_nb_iter_train 7 --dataset cifar10 --model_type resnet_tf
```
