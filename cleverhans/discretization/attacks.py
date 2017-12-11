"""Contains all the attacks on discretized inputs.

The attacks implemented are Discrete Gradient Ascent (DGA) and
Logit Space-Projected Gradient Ascent (LS-PGA).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import discretization_utils


def discretize_range(discretize_fn, levels, low, high, thermometer=False):
  """Get range of discretized values for in the interval (low, high).

    For example, assume discretize_fn uniformly discretizes the values
    between 0 and 1 into 10 bins each represented by either a one hot encoding
    or a thermometer encoding. Then discretize_range(discretize_fn, .3, .7)
    would return [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.]. Note that it's output
    is independent of the encoding used.

  Args:
    discretize_fn: Discretization function to convert into one hot encoding.
    levels: Number of levels to discretize the input into.
    low: Minimum value in the interval.
    high: Maximum value in the interval.
    thermometer: If True, then the discretize_fn returns thermometer codes,
        else it returns one hot codes. (Default: False).

  Returns:
    Mask of 1's over the interval.
  """
  low = tf.clip_by_value(low, 0., 1.)
  high = tf.clip_by_value(high, 0., 1.)
  out = 0.
  for alpha in np.linspace(0., 1., levels):
    q = discretize_fn(alpha * low + (1. - alpha) * high)

    # Convert into one hot encoding if q is in thermometer encoding
    if thermometer:
      q = discretization_utils.thermometer_to_one_hot(q, levels, flattened=True)
    out += q
  return tf.to_float(tf.greater(out, 0.))


def adv_dga(x, model, discretize_fn, projection_fn, levels, phase,
			steps, eps, thermometer=False, noisy_grads=True, y=None):
  """Compute adversarial examples for discretized input using DGA.

  Args:
    x: Input image of shape [-1, height, width, channels] to attack.
    model: Model function which given input returns logits.
    discretize_fn: Function used to discretize the input into one-hot or thermometer
        encoding.
    projection_fn: Function used to project the input before feeding to the
        model (can be identity).
    levels: Number of levels the input has been discretized into.
    phase: Learning phase of the model, corresponding to train and test time.
    steps: Number of steps to iterate when creating adversarial examples.
    eps: Eps ball within which the perturbed image must stay.
    thermometer: Whether the discretized input is in thermometer encoding or one
        hot encoding. (Default: False).
    noisy_grads: If True then compute attack over noisy input.
    y: Optional argument to provide the true labels as opposed to model
        predictions to compute the loss. (Default: None).

  Returns:
    Adversarial image for discretized inputs. The output
    is in the same form of discretization as the input.
  """
  # Add noise
  noise = 0

  if noisy_grads:
    noise = tf.random_uniform(
        shape=tf.shape(x), minval=-eps, maxval=eps, dtype=tf.float32)
  x_noisy = x + noise

  # Clip so that x_noisy is in [0, 1]
  x_noisy = tf.clip_by_value(x_noisy, 0., 1.)

  # Compute the mask over the bits that we are allowed to attack
  mask = discretize_range(
      discretize_fn, levels, x - eps, x + eps, thermometer=thermometer)
  cur_x_discretized = discretize_fn(x_noisy)

  for i in range(steps):
    # Compute one hot representation if input is in thermometer encoding.
    cur_x_one_hot = cur_x_discretized
    if thermometer:
      cur_x_one_hot = discretization_utils.thermometer_to_one_hot(
          cur_x_discretized, levels, flattened=True)

    logits_discretized = model(projection_fn(cur_x_discretized),
                               is_training=phase, reuse=True)

    if i == 0 and y is None:
      # Get one hot version from predictions
      y = tf.one_hot(
          tf.argmax(logits_discretized, 1),
          tf.shape(logits_discretized)[1])

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=logits_discretized)

    # compute the gradients wrt to current input
    grad, = tf.gradients(loss, cur_x_discretized)

    # The harm done by choosing a particular bit to be active
    harm = grad * (1. + cur_x_one_hot - 2 * cur_x_discretized)

    # If we are using thermometer harm is the cumsum
    if thermometer:
        harm_r = discretization_utils.unflatten_last(harm, levels)
        harm_r = tf.cumsum(harm_r, axis=-1, reverse=True)
        harm = discretization_utils.flatten_last(harm_r)

    # Make sure values outside the global mask lose the max
    harm = harm * mask - (1. - mask) * 1000.0

    harm_r = discretization_utils.unflatten_last(harm, levels)

    bit_to_activate = tf.argmax(harm_r, axis=-1)

    one_hot = tf.one_hot(
        bit_to_activate,
        depth=levels,
        on_value=1.,
        off_value=0.,
        dtype=tf.float32,
        axis=-1)

    # Convert into thermometer if we are doing thermometer encodings
    inp = one_hot
    if thermometer:
      inp = discretization_utils.one_hot_to_thermometer(
          one_hot, levels, flattened=False)

    flattened_inp = discretization_utils.flatten_last(inp)
	flattened_inp.mask = mask
    flattened_inp = tf.stop_gradient(flattened_inp)

    cur_x_discretized = flattened_inp
  return flattened_inp


def adv_lspga(x, model, discretize_fn, projection_fn, levels, phase,
              steps, eps, attack_step=1., thermometer=False,
              noisy_grads=True, y=None, inv_temp=1., anneal_rate=1.2):
  """Compute adversarial examples for discretized input by LS-PGA.

  Args:
    x: Input image of shape [-1, height, width, channels] to attack.
    model: Model function which given input returns logits.
    discretize_fn: Function used to discretize the input into one-hot or thermometer
        encoding.
    projection_fn: Function used to project the input before feeding to the
        model (can be identity).
    levels: Number of levels the input has been discretized into.
    phase: Learning phase of the model, corresponding to train and test time.
    steps: Number of steps to iterate when creating adversarial examples.
    eps: Eps ball within which the perturbed image must stay.
    attack_step: Attack step for one iteration of the iterative attack.
    thermometer: Whether the discretized input is in thermometer encoding or one
        hot encoding. (Default: False).
    noisy_grads: If True then compute attack over noisy input.
    y: True labels corresponding to x. If it is None, then use model predictions
        to compute loss, else use true labels. (Default: None).
    inv_temp: Inverse of the temperature parameter for softmax.
    anneal_rate: Rate for annealing the temperature after every iteration of
        attack.

  Returns:
    Adversarial image for discretized inputs. The output
    is in the same form of discretization as the input.
  """
  # Compute the mask over the bits that we are allowed to attack
  flat_mask = discretize_range(
      discretize_fn, levels, x - eps, x + eps, thermometer=thermometer)
  mask = discretization_utils.unflatten_last(flat_mask, levels)
  if noisy_grads:
    activation_logits = tf.random_normal(tf.shape(mask))
  else:
    activation_logits = tf.zeros_like(mask)

  for i in range(steps):
    # Compute one hot representation if input is in thermometer encoding.
    activation_probs = tf.nn.softmax(
        inv_temp * (activation_logits * mask - 999999. * (1. - mask)))

    if thermometer:
      activation_probs = tf.cumsum(activation_probs, axis=-1, reverse=True)

    logits_discretized = model(
        projection_fn(discretization_utils.flatten_last(activation_probs)),
        is_training=phase,
        reuse=True)

    if i == 0 and y is None:
      # Get one hot version from model predictions
      y = tf.one_hot(
          tf.argmax(logits_discretized, 1),
          tf.shape(logits_discretized)[1])

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=logits_discretized)

    # compute the gradients wrt to current logits
    grad, = tf.gradients(loss, activation_logits)

    # Modify activation logits
    activation_logits += attack_step * grad

    # Anneal temperature
    inv_temp *= anneal_rate

  # Convert from logits to actual one-hot image
  final_al = activation_logits * mask - 999999. * (1. - mask)
  bit_to_activate = tf.argmax(final_al, axis=-1)

  one_hot = tf.one_hot(
      bit_to_activate,
      depth=levels,
      on_value=1.,
      off_value=0.,
      dtype=tf.float32,
      axis=-1)

  # Convert into thermometer if we are doing thermometer encodings
  inp = one_hot
  if thermometer:
    inp = discretization_utils.one_hot_to_thermometer(
        one_hot, levels, flattened=False)

  flattened_inp = discretization_utils.flatten_last(inp)

  flattened_inp.mask = mask
  flattened_inp = tf.stop_gradient(flattened_inp)

  return flattened_inp
