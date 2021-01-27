# pylint: disable=missing-docstring
import tensorflow as tf


def preprocess_batch(images_batch, preproc_func=None):
  """
  Creates a preprocessing graph for a batch given a function that processes
  a single image.

  :param images_batch: A tensor for an image batch.
  :param preproc_func: (optional function) A function that takes in a
      tensor and returns a preprocessed input.
  """
  if preproc_func is None:
    return images_batch

  with tf.variable_scope('preprocess'):
    images_list = tf.split(images_batch, int(images_batch.shape[0]))
    result_list = []
    for img in images_list:
      reshaped_img = tf.reshape(img, img.shape[1:])
      processed_img = preproc_func(reshaped_img)
      result_list.append(tf.expand_dims(processed_img, axis=0))
    result_images = tf.concat(result_list, axis=0)
  return result_images
