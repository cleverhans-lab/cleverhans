# MNIST tutorial with the Jacobian-based saliency map attack

This tutorial explains how to use `cleverhans` together 
with a TensorFlow model to craft adversarial examples, 
using the Jacobian-based saliency map approach. This attack
is described in details by the following [paper](https://arxiv.org/abs/1511.07528).
We assume basic knowledge of TensorFlow. If you need help 
getting `cleverhans` installed before getting started, 
you may find our [MNIST tutorial on the fast gradient sign method](mnist_tutorial.md)
to be useful. 

The tutorial's [complete script](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_jsma.py) 
is provided in the `tutorial` folder of the 
`cleverhans` repository. Please be sure to 
add `cleverhans` to your `PYTHONPATH` environment variable
before executing this tutorial. 

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
tf_model_train(sess, x, y, predictions, X_train, Y_train)
```

We can then evaluate the performance of this model
using `tf_model_eval` included in `cleverhans.utils_tf`:

```python
# Evaluate the accuracy of the MNIST model on legitimate test examples
accuracy = tf_model_eval(sess, x, y, predictions, X_test, Y_test)
assert X_test.shape[0] == 10000, X_test.shape
print 'Test accuracy on legitimate test examples: ' + str(accuracy)
```

The accuracy returned should be above `98%`.
The accuracy can become much higher by training for more epochs.

## Crafting adversarial examples - Overview

We first need to create the necessary elements in the TensorFlow graph 
by calling `cleverhans.attacks.jsma` before using the helper
function `cleverhans.utils_tf.batch_eval` to apply it to 
our test set. This gives the following:

```
INSERT UPDATED CODE HERE
```

The second part evaluates the accuracy of the model on 
adversarial examples in a similar way than described 
previously for legitimate examples. It should be
significantly lower than the previous accuracy you obtained on 
legitimate samples from the test set.

Crafting adversarial examples is a 3-step process and is outlined in 
the main loop of the attack, which you may find in the function
`cleverhans.attacks.jsma_tf`:

```python
    # repeat until we have achieved misclassification
    iteration = 0
    current = model_argmax(sess, x, predictions, adv_x)	
    while current != target and iteration < max_iters and len(search_domain) > 0: 

        # compute the Jacobian derivatives
        grads_target, grads_others = jacobian(sess, x, deriv_target, deriv_others, adv_x)

        # compute the salency map for each of our taget classes
        i, j, search_domain = saliency_map(grads_target, grads_others, search_domain, increase)

        # apply an adversarial perterbation to the sample
        adv_x = apply_perturbations(i, j, adv_x, increase, theta, clip_min, clip_max)

        # update our current prediction
        current = model_argmax(sess, x, predictions, adv_x)
        iteration = iteration + 1
```

## Crafting adversarial examples

### The Jacobian

In the first stage of the process, we compute the Jacobian component
corresponding to each pair of output class and input feature. This 
helps us estimating how changes in the input features (here pixels 
of the MNIST images) will affect each of the class probability 
assigned by the model. The Jacobian is a 10 x 28 x 28 matrix of floating 
point numbers where large positive values mean that increasing the 
associated pixels will yield a large increase in the probabilities 
output by the model. Conversely, components with large negative values
correspond to pixels that yield large decreases in the probabilities
output by the model when their value is increased. 
Concisely, this step in the attack is key to identify the features 
we should prioritize when crafting the perturbation that will result
in misclassification.

### The saliency map

In the second stage of the process, we determine the best pixels for 
our adversarial goal: to misclassify an input sample in a chosen target
class by applying the smallest number of perturbations possible to its
input features. To achieve this, we must take into account how much of 
an impact our pixel will have on not only the target class, but all 
other classes as well. Therefore the adversarial saliency score for a 
pixel is defined as the product of the gradient of the target class and 
the sum of the gradients of all other classes (multiplied by -1 so that
we do not consider pixels with a high impact on non-target classes). 

However, this scoring methodology has implications. Ideally, the best 
pixel is one that has a highly positive impact on the target class and a
highly negative impact on all other classes. Nevertheless, such pixels
are rare. In practice, the highest scoring pixels can be placed in one
of two categories; either the pixel has a high positive impact on
the target class and a moderate impact on other classes or the pixel has
little positive impact on our target class, but a highly negative impact
on all other classes.

This would imply then that such a pair of pixels belonging to the two
categories would be ideal in practice as their strengths would cancel
out their weaknesses, i.e. a pixel that has no impact on the target
class but highly negative impact on all other classes should be selected
with a pixel that has a highly positive impact on the target class and
a moderate to low positive impact on all other classes. 

The end result is then a pair of pixels which, when both perturbed 
simultaneously, push us towards our target class while simultaneously 
pushing us away from all other classes. Concisely, it is this pair
of pixels we seek to identify in this step of the attack.

### Applying the perturbations

In the third stage of the process, we simply maximize the value of the 
pixel pair we identified in the prior stage. Depending upon the desired
adversarial example, this functionally means---for this MNIST tutorial
with black and white digits---setting the pixel to absolute black or 
white (0 or 1). Once the  perturbation has been applied, the model is 
queried again to check if we have achieved misclassification. If we have
not successfully perturbed our input sample to our desired target class,
then this process begins again at stage 1 and continues until either we
achieve misclassification, exceeded our maximum desired perturbation 
percentage, or we have exhaustively perturbed all input features (which
should be rare unless the inputs have a small number of features).

## Code

The complete code for this tutorial is available [here](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_jsma.py).
