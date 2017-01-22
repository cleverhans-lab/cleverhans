import theano
import theano.tensor as T
import numpy as np
import time
from cleverhans import utils_th
from cleverhans import utils

def fgsm(x, predictions, eps, clip_min=None, clip_max=None):
    """
    Theano implementation of the Fast Gradient
    Sign method.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :return: a tensor for the adversarial example
    """

    # Compute loss
    y = T.eq(predictions, T.max(predictions, axis=1, keepdims=True))
    y = T.cast(y, utils_th.floatX)
    y = y / T.sum(y, 1, keepdims=True)
    loss = utils_th.model_loss(y, predictions, mean=True)

    # Define gradient of loss wrt input
    grad = T.grad(loss, x)

    # Take sign of gradient
    signed_grad = T.sgn(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = theano.gradient.disconnected_grad(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = T.clip(adv_x, clip_min, clip_max)

    return adv_x


def carlini_L2_step(x, predictions, x0, w, t, beta, eps, kappa, c):
    adv_x = T.nnet.sigmoid(w)
    predictions = theano.clone(predictions, {x: adv_x})
    Z, = predictions.owner.inputs

    inf = 10000
    batch_size = x0.shape[0]

    maxZ = T.max(Z - inf * t, axis=1) - T.sum(Z * t, axis=1)
    f = T.clip(maxZ, -kappa, inf)

    diff = adv_x - x0
    diff = diff.reshape((batch_size, -1))
    norm = diff.norm(L=2, axis=1)

    loss = norm + c * f
    loss = T.mean(loss)

    updates = utils_th.adam(loss, [w], eps)

    import keras
    return theano.function(
        inputs=[],
        outputs=[
            loss,
            norm, f,
            predictions,
            adv_x
        ],
        updates=updates,
        givens={
            keras.backend.learning_phase(): 0,
            x: adv_x
        },
        allow_input_downcast=True,
        rebuild_strict=False
    )


def carlini_L2(x, predictions, samples, labels, beta=0.000001,
                eps=0.01, kappa=0, c=10, nb_iters=50000, batch_size=20):
    """
    Theano implementation of the Carlini L2 method.
    (see https://arxiv.org/abs/1608.04644 for details 
    about the algorithm design choices).
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param samples: the samples to craft adversarial examples for
    :param labels: labels of the samples
    :param beta: coefficient for scaling
    :param eps: the epsilon parameter of Carlini L2 algorithm
    :param kappa: the kappa parameter of Carlini L2 algorithm
    :param c: coefficient for 'f' loss
    :param nb_iters: number of iterations to perform
    :param batch_size: the number of samples in each batch
    :return: crafted adversarial samples
    """
    nb_samples = samples.shape[0]
    sample_shape = samples.shape[1:]
    nb_classes = labels.shape[1]
    batch_size = min(nb_samples, batch_size)
    nb_batches = (nb_samples + batch_size - 1) / batch_size

    assert nb_batches * batch_size >= nb_samples

    batch_samples_shape = (batch_size, ) + sample_shape
    batch_targets_shape = (batch_size, nb_classes)

    x0 = np.zeros(batch_samples_shape, dtype=utils_th.floatX)
    w0 = np.zeros(batch_samples_shape, dtype=utils_th.floatX)
    t0 = np.zeros(batch_targets_shape, dtype=utils_th.floatX)

    x0 = theano.shared(x0, name='x0')
    w = theano.shared(w0, name='w')
    t = theano.shared(t0, name='t')

    
    accuracy = 0.0
    mean_norm = 0.0
    adv_samples = np.zeros_like(samples)

    print("Divided into %d batches." % nb_batches)
    for batch in range(nb_batches):
        start = batch * batch_size
        end = min(start + batch_size, nb_samples)

        batch_samples = samples[start:end]
        batch_labels = labels[start:end]

        x0.set_value(batch_samples)
        
        scaled = batch_samples * (1 - 2 * beta) + beta
        w0 = np.arctanh(2 * scaled - 1)
        w.set_value(w0)

        targets = np.argmax(batch_labels, axis=1) 
        targets += np.random.randint(1, 10, end - start)
        targets %= 10
        t0 = np.eye(10, dtype=utils_th.floatX) [targets]
        t.set_value(t0)

        step = carlini_L2_step(x, predictions, x0, w, t, beta, eps, kappa, c)
        
        print("Batch %2d" % batch)
        prev = time.time()
        for i in range(nb_iters):
            batch_loss, batch_loss_norm, batch_loss_f, pred, batch_adv_samples = step()
            batch_correct = np.argmax(pred, axis=1) == np.argmax(batch_labels, axis=1)
            batch_accuracy = batch_correct.mean()

            if i % 200 == 0:
                print("\tIter %5d: %.3f = %.3f ^ 2 + %.1f * %.5f | accuracy = %.3f" % (
                    i, batch_loss,
                    batch_loss_norm.mean(), c, batch_loss_f.mean(),
                    batch_accuracy
                ))
        cur = time.time()

        print("Batch %2d\taccuracy: %.3f\tL2 norm mean: %.3f | %.3f sec\n" % (
            batch, batch_accuracy, batch_loss_norm.mean(), cur - prev
        ))

        mean_norm += batch_loss_norm.sum()
        accuracy += batch_correct.sum()
        adv_samples[start:end] = batch_adv_samples

    accuracy /= nb_samples
    mean_norm /= nb_samples

    print("Overall accuracy: %.3f\tL2 norm mean: %.3f" % (accuracy, mean_norm))
    return adv_samples
