# Certification of adversarial robustness

This code performs certification of adversarial robustness of given network on
given example.

Original author: [Aditi Raghunathan](https://github.com/RAditi)

## Dependencies

TensorFlow 1.9.0 or higher (to have support of necessary
[autograph](https://www.tensorflow.org/guide/autograph) features)

## Usage

Example usage for two-layer network trained on MNIST:

```bash
# Meaning of the arguments:
# checkpoint - TensorFlow checkpoint of the model parameters
# model_json - JSON file which describes the model (see below)
# test_input - numpy file with input to certify
# true_class - true class of the input
# adv_class - adversarial class for which we're checking certificate,
#             -1 means that all adversarial classes will be checked
# epsilon - maximum size of adversarial perturbation
# init_nu and small_eig_num_steps - optimization parameters
python cleverhans/experimental/certification/certify.py \
  --checkpoint "${MODEL_DIR}/model.ckpt" \
  --model_json "${MODEL_DIR}/model.json" \
  --test_input "${INPUTS_DIR}/image.npy" \
  --true_class 4 \
  --adv_class 5 \
  --epsilon 0.2 \
  --init_nu 100.0 \
  --small_eig_num_steps 100
```

JSON file which accompany checkpoint and describes layers of the network:

```json
[
  {
    "weight_var": "Model/layer0/logits/kernel",
    "bias_var": "Model/layer0/logits/bias",
    "type": "ff_relu",
    "is_transpose": true
  },
  {
    "weight_var": "Model/logits/kernel",
    "bias_var": "Model/logits/bias",
    "type": "ff",
    "is_transpose": true
  }
]
```
