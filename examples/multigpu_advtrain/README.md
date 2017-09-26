This directory includes scripts to train a model using iterative attacks.

The following command trains a model using a single GPU on MNIST:

```python
CUDA_VISIBLE_DEVICES=7 python run.py --save_dir runs/X
```

This command trains a model using multiple GPUs on CIFAR10.

```python
python run_multigpu.py --save_dir runs/X --ngpu 8 --fast_tests --attack_nb_iter_train 7 --dataset cifar10 --model_type resnet_tf
```
