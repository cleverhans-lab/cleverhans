Try these commands:

```python
CUDA_VISIBLE_DEVICES=7 python run.py --save_dir runs/X
python run_multigpu.py --save_dir runs/X --ngpu 8 --fast_tests --attack_nb_iter_train 7 --dataset cifar10 --model_type resnet_tf
```
