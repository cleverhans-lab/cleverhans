"""File containing some simple helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def diag(diag_elements):
  """Function to create tensorflow diagonal matrix with input diagonal entries.

  Args:
    diag_elements: tensor with diagonal elements

  Returns:
    tf matrix with diagonal entries as diag_elements
  """
  return tf.diag(tf.reshape(diag_elements, [-1]))

def eig_one_step(current_vector, learning_rate, vector_prod_fn):
  """Function that performs one step of gd (variant) for min eigen value.

  Args:
    current_vector: current estimate of the eigen vector with minimum eigen
      value.
    learning_rate: learning rate.
    vector_prod_fn: function which returns product H*x, where H is a matrix for
      which we computing eigenvector.

  Returns:
    updated vector after one step
  """
  grad = 2*vector_prod_fn(current_vector)
  # Current objective = (1/2)*v^T (2*M*v); v = current_vector
  # grad = 2*M*v
  current_objective = tf.reshape(tf.matmul(tf.transpose(current_vector),
                                           grad) / 2., shape=())

  # Project the gradient into the tangent space of the constraint region.
  # This way we do not waste time taking steps that try to change the
  # norm of current_vector
  grad = grad - current_vector*tf.matmul(tf.transpose(current_vector), grad)
  grad_norm = tf.norm(grad)
  grad_norm_sq = tf.square(grad_norm)

  # Computing normalized gradient of unit norm
  norm_grad = grad / grad_norm

  # Computing directional second derivative (dsd)
  # dsd = 2*g^T M g, where g is normalized gradient
  directional_second_derivative = (
      tf.reshape(2*tf.matmul(tf.transpose(norm_grad),
                             vector_prod_fn(norm_grad)),
                 shape=()))

  # Computing grad^\top M grad [useful to compute step size later]
  # Just a rescaling of the directional_second_derivative (which uses
  # normalized gradient
  grad_m_grad = directional_second_derivative*grad_norm_sq / 2

  # Directional_second_derivative/2 = objective when vector is norm_grad
  # If this is smaller than current objective, simply return that
  if directional_second_derivative / 2. < current_objective:
    return norm_grad

  # If curvature is positive, jump to the bottom of the bowl
  if directional_second_derivative > 0.:
    step = -1. * grad_norm / directional_second_derivative
  else:
    # If the gradient is very small, do not move
    if grad_norm_sq <= 1e-16:
      step = 0.0
    else:
      # Make a heuristic guess of the step size
      step = -2. * tf.reduce_sum(current_vector*grad) / grad_norm_sq
      # Computing gain using the gradient and second derivative
      gain = -(2 * tf.reduce_sum(current_vector*grad) +
               (step*step) * grad_m_grad)

      # Fall back to pre-determined learning rate if no gain
      if gain < 0.:
        step = -learning_rate * grad_norm
  current_vector = current_vector + step * norm_grad
  return tf.nn.l2_normalize(current_vector)


def minimum_eigen_vector(x, num_steps, learning_rate, vector_prod_fn):
  """Computes eigenvector which corresponds to minimum eigenvalue.

  Args:
    x: initial value of eigenvector.
    num_steps: number of optimization steps.
    learning_rate: learning rate.
    vector_prod_fn: function which takes x and returns product H*x.

  Returns:
    approximate value of eigenvector.

  This function finds approximate value of eigenvector of matrix H which
  corresponds to smallest (by absolute value) eigenvalue of H.
  It works by solving optimization problem x^{T}*H*x -> min.
  """
  x = tf.nn.l2_normalize(x)
  for _ in range(num_steps):
    x = eig_one_step(x, learning_rate, vector_prod_fn)
  return x
