# Examples

* **CIFAR10 with FGSM using the TensorFlow backend** ([code](ex_cifar10_tf.py)): this tutorial covers how to train a CIFAR10 model using TensorFlow,
craft adversarial examples using the [fast gradient sign method](https://arxiv.org/abs/1412.6572), 
and make the model more robust to adversarial
examples using adversarial training.
* **Adversarial faces using FGSM against Facenet library** ([code](facenet_adversarial_faces)): this example covers how to create adversarial faces using the [fast gradient sign method](https://arxiv.org/abs/1412.6572) over the Facenet library.

# Additional dependencies

Besides the standard CleverHans dependencies, these examples also require
`wget`.
