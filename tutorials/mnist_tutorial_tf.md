# MNIST tutorial: the fast gradient sign method and adversarial training

This tutorial explains how to use `cleverhans` together
with a TensorFlow model to craft adversarial examples,
as well as make the model more robust to adversarial
examples. We assume basic knowledge of TensorFlow.

## Setup

First, make sure that you have [TensorFlow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#download-and-setup)
and [Keras](https://keras.io/#installation) installed on
your machine and then clone the `cleverhans`
[repository](https://github.com/openai/cleverhans).
Also, add the path of the repository clone to your
`PYTHONPATH` environment variable.
```bash
export PYTHONPATH="/path/to/cleverhans":$PYTHONPATH
```
This allows our tutorial script to import the library
simply with `import cleverhans`.

The tutorial's [complete script](mnist_tutorial_tf.py)
is provided in the `tutorial` folder of the
`cleverhans` repository.

## Defining the model with TensorFlow and Keras

In this tutorial, we use Keras to define the model
and TensorFlow to train it. The model is a Keras
[`Sequential` model](https://keras.io/models/sequential/):
it is made up of multiple convolutional and ReLU layers.
You can find the model definition in the
[`utils_mnist` cleverhans module](https://github.com/openai/cleverhans/blob/master/cleverhans/utils_mnist.py).

```python
# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
y = tf.placeholder(tf.float32, shape=(None, FLAGS.nb_classes))

# Define TF model graph
model = model_mnist()
predictions = model(x)
print "Defined TensorFlow model graph."
```

## Training the model with TensorFlow

The library includes a helper function that runs a
TensorFlow optimizer to train models and another
helper function to load the MNIST dataset.
To train our MNIST model, we run the following:

```python
# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist()

# Train an MNIST model
model_train(sess, x, y, predictions, X_train, Y_train)
```

We can then evaluate the performance of this model
using `model_eval` included in `cleverhans.utils_tf`:

```python
# Evaluate the accuracy of the MNIST model on legitimate test examples
accuracy = model_eval(sess, x, y, predictions, X_test, Y_test)
assert X_test.shape[0] == 10000, X_test.shape
print 'Test accuracy on legitimate test examples: ' + str(accuracy)
```

The accuracy returned should be above `98%`.
The accuracy can become much higher by training for more epochs.

## Crafting adversarial examples

This tutorial applies the Fast Gradient Sign method
introduced by [Goodfellow et al.](https://arxiv.org/abs/1412.6572).
We first need to create the necessary graph elements by
calling `cleverhans.attacks.fgsm` before using the helper
function `cleverhans.utils_tf.batch_eval` to apply it to
our test set. This gives the following:

```python
# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
adv_x = fgsm(x, predictions, eps=0.3)
X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
assert X_test_adv.shape[0] == 10000, X_test_adv.shape

# Evaluate the accuracy of the MNIST model on adversarial examples
accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test)
print'Test accuracy on adversarial examples: ' + str(accuracy)
```

The second part evaluates the accuracy of the model on
adversarial examples in a similar way than described
previously for legitimate examples. It should be
significantly lower than the previous accuracy you obtained.


## Improving robustness using adversarial training

One defense strategy to mitigate adversarial examples is to use
adversarial training, i.e. train the model with both the
original data and adversarially modified data (with correct
labels). You can use the training function `utils_tf.model_train`
with the optional argument `predictions_adv` set to the result
of `cleverhans.attacks.fgsm` in order to perform adversarial
training.

In the following snippet, we first declare a new model (in a
way similar to the one described previously) and then we train
it with both legitimate and adversarial training points.

```python
# Redefine TF model graph
model_2 = model_mnist()
predictions_2 = model_2(x)
adv_x_2 = fgsm(x, predictions_2, eps=0.3)
predictions_2_adv = model_2(adv_x_2)

# Perform adversarial training
model_train(sess, x, y, predictions_2, X_train, Y_train, predictions_adv=predictions_2_adv)
```

We can then verify that (1) its accuracy on legitimate data is
still comparable to the first model, (2) its accuracy on newly
generated adversarial examples is higher.

```python
# Evaluate the accuracy of the adversarialy trained MNIST model on
# legitimate test examples
accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test)
print 'Test accuracy on legitimate test examples: ' + str(accuracy)

# Craft adversarial examples using Fast Gradient Sign Method (FGSM) on
# the new model, which was trained using adversarial training
X_test_adv_2, = batch_eval(sess, [x], [adv_x_2], [X_test])
assert X_test_adv_2.shape[0] == 10000, X_test_adv_2.shape

# Evaluate the accuracy of the adversarially trained MNIST model on
# adversarial examples
accuracy_adv = model_eval(sess, x, y, predictions_2, X_test_adv_2, Y_test)
print'Test accuracy on adversarial examples: ' + str(accuracy_adv)
```

## Code

The complete code for this tutorial is available [here](mnist_tutorial_tf.py).
