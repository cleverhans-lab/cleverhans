# MNIST tutorial: crafting adversarial examples with the Jacobian-based saliency map attack

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
y = tf.placeholder(tf.float32, shape=(None, 10))

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

We first need to create the elements in the TensorFlow graph necessary
to compute Jacobian matrices (see below for more details), as well as
two numpy arrays to keep track of the results of adversarial example
crafting.

```python
# This array indicates whether an adversarial example was found for each
# test set sample and target class
results = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='i')

# This array contains the fraction of perturbed features for each test set
# sample and target class
perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='f')

# Define the TF graph for the model's Jacobian
grads = jacobian_graph(predictions, x)
```

We then iterate over the samples that we want to perturb and all
possible target classes (i.e. all classes that are different from
the label assigned to the input in the dataset).

```python
# Loop over the samples we want to perturb into adversarial examples
for sample_ind in xrange(FLAGS.source_samples):
    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset)
    target_classes = list(xrange(FLAGS.nb_classes))
    target_classes.remove(int(np.argmax(Y_test[sample_ind])))

    # Loop over all target classes
    for target in target_classes:
        print('--------------------------------------')
        print('Creating adversarial example for target class ' + str(target))

        # This call runs the Jacobian-based saliency map approach
        _, result, percentage_perturb = jsma(sess, x, predictions, grads,
                                             X_test[sample_ind:(sample_ind+1)],
                                             target, theta=1, gamma=0.1,
                                             increase=True, back='tf',
                                             clip_min=0, clip_max=1)

        # Update the arrays for later analysis
        results[target, sample_ind] = result
        perturbations[target, sample_ind] = percentage_perturb
```

The last few lines analyze the numpy arrays updated throughout crafting
in order to compute the success rate of the adversary: the number of
source-target misclassifications that were successful. Therefore, the
adversarial success rate is the opposite of the model's accuracy.
Given that the adversarial success rate should be larger than `90%`,
the model's accuracy on these adversarial examples is lower than `10%`.
This should be
significantly lower than the previous accuracy you obtained on
legitimate samples from the test set.
It also provides
the average fraction of input features perturbed to achieve this
misclassification.

### Overview of the crafting process

Crafting adversarial examples is a 3-step process and is outlined in
the main loop of the attack, which you may find in the function
`cleverhans.attacks.jsma_tf`:

```python
# Compute the Jacobian components
grads_target, grads_others = jacobian(sess, x, grads, target, adv_x)

# Compute the saliency map for each of our target classes
# and return the two best candidate features for perturbation
i, j, search_domain = saliency_map(grads_target, grads_others, search_domain, increase)

# Apply the perturbation to the two input features selected previously
adv_x = apply_perturbations(i, j, adv_x, increase, theta, clip_min, clip_max)
```

### The Jacobian

In the first stage of the process, we compute the Jacobian component
corresponding to each pair of output class and input feature. This
helps us estimate how changes in the input features (here pixels
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

Before computing its actual values, the Jacobian is defined as a TF
graph by one call to `attacks.jacobian_graph()`. This graph is ran
by function `attacks.jacobian()`, which is fed with the current
values of input features to be fed as the input of the graph.

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

The computation of saliency map scores for pixel pairs is defined in
function `attacks.saliency_score()`. It is used to compute the entire
saliency map of an input with a pool
of threads by function `attacks.saliency_map()`.

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

This perturbation is applied by function `attacks.apply_perturbations`,
which makes sure that the resulting adversarial example remains in the
expected input domain (i.e., it constraints the perturbed input features
to remain between 0 and 1 in the case of MNIST).

## Code

The complete code for this tutorial is available [here](https://github.com/openai/cleverhans/blob/master/tutorials/mnist_tutorial_jsma.py).
