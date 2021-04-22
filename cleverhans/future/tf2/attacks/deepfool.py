""" The DeepFool attack """

import copy
import numpy as np
import tensorflow as tf


def deepfool(image_norm, model, num_classes=10, overshoot=0.02, max_iter=50):
  """
  DeepFool is an untargeted & iterative attack which is based on an
  iterative linearization of the classifier. The implementation here
  is w.r.t. the L2 norm.
  Paper link: "https://arxiv.org/pdf/1511.04599.pdf"

  Tensorflow 2.0 implementation of the DeepFool Method.

  :param image_norm: [batch, width, height, channels], like [1, 28, 28, 1].
  :param model: network (input: images, output: values of activation **BEFORE** softmax).
  :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
  :param max_iter: maximum number of iterations for deepfool (default = 50)

  :return: minimal perturbation that fools the classifier, number of iterations that it required,
          new estimated_label and perturbed image
  """

  f_image = model(image_norm).numpy().flatten()
  I = (np.array(f_image)).flatten().argsort()[::-1]
  I = I[0:num_classes]
  label = I[0]

  input_shape = np.shape(image_norm)
  pert_image = copy.deepcopy(image_norm)
  w = np.zeros(input_shape)
  r_tot = np.zeros(input_shape)

  loop_i = 0
  x = tf.Variable(pert_image)
  fs = model(x)  # shape=(1, num_classes)
  k_i = label

  def loss_func(logits, I, k):
    return logits[0, I[k]]

  while k_i == label and loop_i < max_iter:
    pert = np.inf
    one_hot_label_0 = tf.one_hot(label, num_classes)
    with tf.GradientTape() as tape:
      tape.watch(x)
      fs = model(x)
      loss_value = loss_func(fs, I, 0)
    grad_orig = tape.gradient(loss_value, x)

    for k in range(1, num_classes):
      one_hot_label_k = tf.one_hot(I[k], num_classes)
      with tf.GradientTape() as tape:
        tape.watch(x)
        fs = model(x)
        loss_value = loss_func(fs, I, k)
      cur_grad = tape.gradient(loss_value, x)

      w_k = cur_grad - grad_orig

      f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

      pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, [-1]))

      if pert_k < pert:
        pert = pert_k
        w = w_k

    r_i = (pert + 1e-4) * w / np.linalg.norm(w)
    r_tot = np.float32(r_tot + r_i)
    pert_image = image_norm + (1 + overshoot) * r_tot

    x = tf.Variable(pert_image)

    fs = model(x)
    k_i = np.argmax(np.array(fs).flatten())

    loop_i += 1

  r_tot = (1 + overshoot) * r_tot

  return r_tot, loop_i, label, k_i, pert_image
