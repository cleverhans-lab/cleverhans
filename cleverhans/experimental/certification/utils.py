"""File containing some simple helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def diag(diag_elements):
  """Function to create tensorflow diagonal matrix with input diagonal entries.

  Args:
    diag_elements: tensor with diagonal elements

  Returns:
    tf matrix with diagonal entries as diag_elements
  """
  return tf.diag(tf.reshape(diag_elements, [-1]))


def initialize_dual(neural_net_params_object, init_dual_file=None,
                    random_init_variance=0.01, init_nu=200.0):
  """Function to initialize the dual variables of the class.

  Args:
    neural_net_params_object: Object with the neural net weights, biases
      and types
    init_dual_file: Path to file containing dual variables, if the path
      is empty, perform random initialization
      Expects numpy dictionary with
      lambda_pos_0, lambda_pos_1, ..
      lambda_neg_0, lambda_neg_1, ..
      lambda_quad_0, lambda_quad_1, ..
      lambda_lu_0, lambda_lu_1, ..
      random_init_variance: variance for random initialization
    init_nu: Value to initialize nu variable with

  Returns:
    dual_var: dual variables initialized appropriately.
  """
  lambda_pos = []
  lambda_neg = []
  lambda_quad = []
  lambda_lu = []

  if init_dual_file is None:
    for i in range(0, neural_net_params_object.num_hidden_layers + 1):
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_pos.append(tf.get_variable('lambda_pos_' + str(i),
                                        initializer=initializer,
                                        dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_neg.append(tf.get_variable('lambda_neg_' + str(i),
                                        initializer=initializer,
                                        dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_quad.append(tf.get_variable('lambda_quad_' + str(i),
                                         initializer=initializer,
                                         dtype=tf.float32))
      initializer = (np.random.uniform(0, random_init_variance, size=(
          neural_net_params_object.sizes[i], 1))).astype(np.float32)
      lambda_lu.append(tf.get_variable('lambda_lu_' + str(i),
                                       initializer=initializer,
                                       dtype=tf.float32))
    nu = tf.get_variable('nu', initializer=init_nu)
  else:
    # Loading from file
    dual_var_init_val = np.load(init_dual_file).item()
    for i in range(0, neural_net_params_object.num_hidden_layers + 1):
      lambda_pos.append(
          tf.get_variable('lambda_pos_' + str(i),
                          initializer=dual_var_init_val['lambda_pos'][i],
                          dtype=tf.float32))
      lambda_neg.append(
          tf.get_variable('lambda_neg_' + str(i),
                          initializer=dual_var_init_val['lambda_neg'][i],
                          dtype=tf.float32))
      lambda_quad.append(
          tf.get_variable('lambda_quad_' + str(i),
                          initializer=dual_var_init_val['lambda_quad'][i],
                          dtype=tf.float32))
      lambda_lu.append(
          tf.get_variable('lambda_lu_' + str(i),
                          initializer=dual_var_init_val['lambda_lu'][i],
                          dtype=tf.float32))
    nu = tf.get_variable('nu', initializer=1.0*dual_var_init_val['nu'])
  dual_var = {'lambda_pos': lambda_pos, 'lambda_neg': lambda_neg,
              'lambda_quad': lambda_quad, 'lambda_lu': lambda_lu, 'nu': nu}
  return dual_var

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


def tf_lanczos_smallest_eigval(vector_prod_fn,
                               matrix_dim,
                               initial_vector,
                               num_iter=1000,
                               max_iter=1000,
                               collapse_tol=1e-9,
                               dtype=tf.float32):
  """Computes smallest eigenvector and eigenvalue using Lanczos in pure TF.

  This function computes smallest eigenvector and eigenvalue of the matrix
  which is implicitly specified by `vector_prod_fn`.
  `vector_prod_fn` is a function which takes `x` and returns a product of matrix
  in consideration and `x`.
  Computation is done using Lanczos algorithm, see
  https://en.wikipedia.org/wiki/Lanczos_algorithm#The_algorithm

  Args:
    vector_prod_fn: function which takes a vector as an input and returns
      matrix vector product.
    matrix_dim: dimentionality of the matrix.
    initial_vector: guess vector to start the algorithm with
    num_iter: user-defined number of iterations for the algorithm
    max_iter: maximum number of iterations.
    collapse_tol: tolerance to determine collapse of the Krylov subspace
    dtype: type of data

  Returns:
    tuple of (eigenvalue, eigenvector) of smallest eigenvalue and corresponding
    eigenvector.
  """

  # alpha will store diagonal elements
  alpha = tf.TensorArray(dtype, size=1, dynamic_size=True, element_shape=())
  # beta will store off diagonal elements
  beta = tf.TensorArray(dtype, size=0, dynamic_size=True, element_shape=())
  # q will store Krylov space basis
  q_vectors = tf.TensorArray(
      dtype, size=1, dynamic_size=True, element_shape=(matrix_dim, 1))

  # If start vector is all zeros, make it a random normal vector and run for max_iter
  if tf.norm(initial_vector) < collapse_tol:
    initial_vector = tf.random_normal(shape=(matrix_dim, 1), dtype=dtype)
    num_iter = max_iter

  w = initial_vector / tf.norm(initial_vector)

  # Iteration 0 of Lanczos
  q_vectors = q_vectors.write(0, w)
  w_ = vector_prod_fn(w)
  cur_alpha = tf.reduce_sum(w_ * w)
  alpha = alpha.write(0, cur_alpha)
  w_ = w_ - tf.scalar_mul(cur_alpha, w)
  w_prev = w
  w = w_

  # Subsequent iterations of Lanczos
  for i in tf.range(1, num_iter):
    cur_beta = tf.norm(w)
    if cur_beta < collapse_tol:
      # return early if Krylov subspace collapsed
      break

    # cur_beta is larger than collapse_tol,
    # so division will return finite result.
    w = w / cur_beta

    w_ = vector_prod_fn(w)
    cur_alpha = tf.reduce_sum(w_ * w)

    q_vectors = q_vectors.write(i, w)
    alpha = alpha.write(i, cur_alpha)
    beta = beta.write(i-1, cur_beta)

    w_ = w_ - tf.scalar_mul(cur_alpha, w) - tf.scalar_mul(cur_beta, w_prev)
    w_prev = w
    w = w_

  alpha = alpha.stack()
  beta = beta.stack()
  q_vectors = tf.reshape(q_vectors.stack(), (-1, matrix_dim))

  offdiag_submatrix = tf.linalg.diag(beta)
  tridiag_matrix = (tf.linalg.diag(alpha)
                    + tf.pad(offdiag_submatrix, [[0, 1], [1, 0]])
                    + tf.pad(offdiag_submatrix, [[1, 0], [0, 1]]))

  eigvals, eigvecs = tf.linalg.eigh(tridiag_matrix)

  smallest_eigval = eigvals[0]
  smallest_eigvec = tf.matmul(tf.reshape(eigvecs[:, 0], (1, -1)),
                              q_vectors)
  smallest_eigvec = smallest_eigvec / tf.norm(smallest_eigvec)
  smallest_eigvec = tf.reshape(smallest_eigvec, (matrix_dim, 1))

  return smallest_eigval, smallest_eigvec
