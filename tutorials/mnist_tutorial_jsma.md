# MNIST tutorial

This tutorial explains how to use `cleverhans` together 
with a TensorFlow model to craft adversarial examples, 
using the Jacobian-based forward derivative.
We assume basic knowledge of TensorFlow. 

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

The tutorial's [complete script](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_jsma.py) 
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

This tutorial applies the Jacobian-based forwad derivative method
introduced by [this paper](https://arxiv.org/abs/1511.07528).
We first need to create the necessary graph elements by 
calling `cleverhans.attacks.jsma` before using the helper
function `cleverhans.utils_tf.batch_eval` to apply it to 
our test set. This gives the following:

```python
    #Craft adversarial examples for nb_classes from per_samples using the Jacobian-based saliency map approach 
    results = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='i')
    perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='f')
    print 'Crafting ' + str(FLAGS.source_samples) + ' * ' + str(FLAGS.nb_classes) + ' adversarial examples'

    for sample in xrange(FLAGS.source_samples):
        target_classes = list(xrange(FLAGS.nb_classes))
        target_classes.remove(np.argmax(Y_test[sample]))
        for target in target_classes:
            print '--------------------------------------\nCreating adversarial example for target class ' + str(target)
            _, result, percentage_perterb = jsma(sess, x, predictions, X_test[sample:(sample+1)], target,
                    theta=1, gamma=0.1, increase=True, back='tf', clip_min=0, clip_max=1)
            results[target, sample] = result
            perturbations[target, sample] = percentage_perterb

# Evaluate the accuracy of the MNIST model on adversarial examples
accuracy = tf_model_eval(sess, x, y, predictions, X_test_adv, Y_test)
print'Test accuracy on adversarial examples: ' + str(accuracy)
```

The second part evaluates the accuracy of the model on 
adversarial examples in a similar way than described 
previously for legitimate examples. It should be
significantly lower than the previous accuracy you obtained.

Crafting adversarial examples is a 3-step process and is outlined in 
the main loop of the attack:

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

## Crafting adversarial examples - the Jacobian

In the first stage of the process, we compute the Jacobian forward derivatives for
each class to determine how changes in the pixels will affect the classification 
of the input sample. These changes are represented as 28 x 28 floating point numbers
where higher values mean the associated pixels have a large influence in misclassifying
the input sample to a target class (Conversely, negative values imply a negligable impact). 
Concisely, this step in the attack is to identify the features we might be interested in perterbing.

## Crafting adversarial examples - the Salency map

In the second stage of the process, we determine the best pixes for our goal: to 
misclassify an input sample by applying the smallest number of perterbations possible.
To achieve this, we must take into account how much of an impact our pixel will have on
not only the target class, but all other classes as well. Ideally, we would like our model
to confidently misclassify our input sample, therefore the score for a pixel is defined as
the product of the gradient of the target class and the sum of the gradients of all other 
classes (multiplied by -1 so that we do not consider pixels that have a high impact on 
non-target classes). 

However, this scoring methodology has implications. Ideally, the best pixel is one that
has a highly positive impact on the target class and a highly negative impact on all 
other classes. Nevertheless, such pixels are rare. In practice, the highest scoring pixels
can be placed in one of two categories; either the pixel has a high positive impact on
the target class and a moderate impact on other classes or the pixel has little positive impact
on our target class, but a highly negative impact on all other classes.

This would imply then that such a pair of pixels belonging to the two categories would be 
ideal in practice as their strenghts would cancel out their weaknesses, i.e. a pixel
that has no impact on the target class but highly negative impact on all other classes
should be selected with a pixel that has a highly positive impact on the target class and
a moderate to low positive impact on all other classes. 

The end result is then a pair of pixels who, when both perterbed, push us towards our target
class while simultaneously push us away from all other classes. Concisely, it is this pair
of pixels we seek to identify in this step of the attack.

## Crafting adversarial examples - applying the perterbations

In the third stage of the process, we simply maximize the value of the pixel pair we 
identified in the prior stage. Depending upon the desired adversarial example, this 
functionally means setting the pixel to absolute black or white (0 or 1). Once the 
perterbation has been applied, the model is queried again to check if we have 
achieved misclassification. If we have not succesfully perterbed our input sample
to our desired target class, then this process begins again at stage 1 and continues
until either we achieve misclassification, exceeded our maximum desired perterbation 
percentage, or we have exhaustively perterbed all input features (which should never
happen unless your input samples have an incredibly low number of features).

## Code

The complete code for this tutorial is available [here](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_jsma.py).
