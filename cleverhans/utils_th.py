from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np
import os
import six
import time

from collections import OrderedDict

from .utils import batch_indices, _ArgsWrapper

import theano
import theano.tensor as T

import keras

floatX = theano.config.floatX

_TEST_PHASE = np.uint8(0)
_TRAIN_PHASE = np.uint8(1)


def get_or_compute_grads(loss_or_grads, params):
    if isinstance(loss_or_grads, list):
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)


def adadelta(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    """ From Lasagne
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


def model_loss(y, model, mean=True):
    """
    Define loss of Theano graph
    :param y: correct labels
    :param model: output of the model
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    from_logits = "softmax" not in str(model).lower()

    if from_logits:
        model = T.nnet.softmax(model)

    out = T.nnet.categorical_crossentropy(model, y)

    if mean:
        out = T.mean(out)
    return out


def th_model_train(x, y, predictions, params, X_train, Y_train, save=False,
                   predictions_adv=None, evaluate=None, args=None):
    """
    Train a Theano graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param params: model trainable weights
    :param X_train: numpy array with training inputs
    :param Y_train: numpy array with training outputs
    :param save: boolean controling the save operation
    :param predictions_adv: if set with the adversarial example tensor,
                            will run adversarial training
    :param args: dict or argparse `Namespace` object.
                 Should contain `nb_epochs`, `learning_rate`,
                 `batch_size`
    :return: True if model trained
    """
    args = _ArgsWrapper(args or {})

    print("Starting model training using Theano.")

    # Define loss
    loss = model_loss(y, predictions)
    if predictions_adv is not None:
        loss = (loss + model_loss(y, predictions_adv)) / 2

    print("Defined optimizer.")

    train_step = theano.function(
        inputs=[x, y],
        outputs=[loss],
        givens={keras.backend.learning_phase(): _TRAIN_PHASE},
        allow_input_downcast=True,
        updates=adadelta(
            loss, params, learning_rate=args.learning_rate, rho=0.95,
            epsilon=1e-08)
    )

    for epoch in six.moves.xrange(args.nb_epochs):
        print("Epoch " + str(epoch))

        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_train)

        prev = time.time()
        for batch in range(nb_batches):

            # Compute batch start and end indices
            start, end = batch_indices(batch, len(X_train), args.batch_size)

            # Perform one training step
            train_step(X_train[start:end], Y_train[start:end])
        assert end >= len(X_train)  # Check that all examples were used
        cur = time.time()
        print("\tEpoch took " + str(cur - prev) + " seconds")
        prev = cur
        if evaluate is not None:
            evaluate()

    return True


def th_model_eval(x, y, model, X_test, Y_test, args=None):
    """
    Compute the accuracy of a Theano model on some data
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    args = _ArgsWrapper(args or {})

    # Define symbol for accuracy
    acc_value = keras.metrics.categorical_accuracy(y, model)

    # Init result var
    accuracy = 0.0

    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)

    eval_step = theano.function(
        inputs=[x, y],
        outputs=acc_value,
        givens={keras.backend.learning_phase(): _TEST_PHASE},
        allow_input_downcast=True,
        updates=None
    )

    for batch in range(nb_batches):
        if batch % 100 == 0 and batch > 0:
            print("Batch " + str(batch))

        # Must not use the `batch_indices` function here, because it
        # repeats some examples.
        # It's acceptable to repeat during training, but not eval.
        start = batch * args.batch_size
        end = min(len(X_test), start + args.batch_size)
        cur_batch_size = end - start

        # The last batch may be smaller than all others, so we need to
        # account for variable batch size here
        accuracy += cur_batch_size * \
            eval_step(X_test[start:end], Y_test[start:end])
    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)

    return accuracy


def batch_eval(th_inputs, th_outputs, numpy_inputs, args=None):
    """
    A helper function that computes a tensor on numpy inputs by batches.

    :param th_inputs:
    :param th_outputs:
    :param numpy_inputs:
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    """
    args = _ArgsWrapper(args or {})

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(th_inputs)
    m = numpy_inputs[0].shape[0]
    for i in six.moves.xrange(1, n):
        assert numpy_inputs[i].shape[0] == m
    out = []
    for _ in th_outputs:
        out.append([])

    eval_step = theano.function(
        inputs=th_inputs,
        outputs=th_outputs,
        givens={keras.backend.learning_phase(): _TEST_PHASE},
        allow_input_downcast=True,
        updates=None
    )

    for start in six.moves.xrange(0, m, args.batch_size):
        batch = start // args.batch_size
        if batch % 100 == 0 and batch > 0:
            print("Batch " + str(batch))

        # Compute batch start and end indices
        start = batch * args.batch_size
        end = start + args.batch_size
        numpy_input_batches = [numpy_input[start:end]
                               for numpy_input in numpy_inputs]
        cur_batch_size = numpy_input_batches[0].shape[0]
        assert cur_batch_size <= args.batch_size
        for e in numpy_input_batches:
            assert e.shape[0] == cur_batch_size

        numpy_output_batches = eval_step(*numpy_input_batches)
        for e in numpy_output_batches:
            assert e.shape[0] == cur_batch_size, e.shape
        for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
            out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def model_argmax(x, predictions, sample):
    """
    Helper function that computes the current class prediction
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param sample: (1 x 1 x img_rows x img_cols) numpy array with sample input
    :return: the argmax output of predictions, i.e. the current predicted class
    """

    probabilities = theano.function(
        inputs=[x],
        outputs=predictions,
        givens={keras.backend.learning_phase(): _TEST_PHASE},
        allow_input_downcast=True,
        updates=None
    )(x)

    return np.argmax(probabilities)
