"""
Functionality for showing images in pyplot.
See also cleverhans.plot.image for loading/saving image files, showing
images in 3rd party viewers, etc.
"""
import numpy as np
from six.moves import range

def pair_visual(original, adversarial, figure=None):
  """
  This function displays two images: the original and the adversarial sample
  :param original: the original input
  :param adversarial: the input after perturbations have been applied
  :param figure: if we've already displayed images, use the same plot
  :return: the matplot figure to reuse for future samples
  """
  import matplotlib.pyplot as plt

  # Squeeze the image to remove single-dimensional entries from array shape
  original = np.squeeze(original)
  adversarial = np.squeeze(adversarial)

  # Ensure our inputs are of proper shape
  assert(len(original.shape) == 2 or len(original.shape) == 3)

  # To avoid creating figures per input sample, reuse the sample plot
  if figure is None:
    plt.ion()
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Pair Visualization')

  # Add the images to the plot
  perturbations = adversarial - original
  for index, image in enumerate((original, perturbations, adversarial)):
    figure.add_subplot(1, 3, index + 1)
    plt.axis('off')

    # If the image is 2D, then we have 1 color channel
    if len(image.shape) == 2:
      plt.imshow(image, cmap='gray')
    else:
      plt.imshow(image)

    # Give the plot some time to update
    plt.pause(0.01)

  # Draw the plot and return
  plt.show()
  return figure

def grid_visual(data):
  """
  This function displays a grid of images to show full misclassification
  :param data: grid data of the form;
      [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
  :return: if necessary, the matplot figure to reuse
  """
  import matplotlib.pyplot as plt

  # Ensure interactive mode is disabled and initialize our graph
  plt.ioff()
  figure = plt.figure()
  figure.canvas.set_window_title('Cleverhans: Grid Visualization')

  # Add the images to the plot
  num_cols = data.shape[0]
  num_rows = data.shape[1]
  num_channels = data.shape[4]
  for y in range(num_rows):
    for x in range(num_cols):
      figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
      plt.axis('off')

      if num_channels == 1:
        plt.imshow(data[x, y, :, :, 0], cmap='gray')
      else:
        plt.imshow(data[x, y, :, :, :])

  # Draw the plot and return
  plt.show()
  return figure


def get_logits_over_interval(sess, model, x_data, fgsm_params,
                             min_epsilon=-10., max_epsilon=10.,
                             num_points=21):
  """Get logits when the input is perturbed in an interval in adv direction.

  Args:
      sess: Tf session
      model: Model for which we wish to get logits.
      x_data: Numpy array corresponding to single data.
              point of shape [height, width, channels].
      fgsm_params: Parameters for generating adversarial examples.
      min_epsilon: Minimum value of epsilon over the interval.
      max_epsilon: Maximum value of epsilon over the interval.
      num_points: Number of points used to interpolate.

  Returns:
      Numpy array containing logits.

  Raises:
      ValueError if min_epsilon is larger than max_epsilon.
  """
  # Get the height, width and number of channels
  height = x_data.shape[0]
  width = x_data.shape[1]
  channels = x_data.shape[2]

  x_data = np.expand_dims(x_data, axis=0)
  import tensorflow as tf
  from cleverhans.attacks import FastGradientMethod

  # Define the data placeholder
  x = tf.placeholder(dtype=tf.float32,
                     shape=[1, height,
                            width,
                            channels],
                     name='x')
  # Define adv_x
  fgsm = FastGradientMethod(model, sess=sess)
  adv_x = fgsm.generate(x, **fgsm_params)

  if min_epsilon > max_epsilon:
    raise ValueError('Minimum epsilon is less than maximum epsilon')

  eta = tf.nn.l2_normalize(adv_x - x, dim=0)
  epsilon = tf.reshape(tf.lin_space(float(min_epsilon),
                                    float(max_epsilon),
                                    num_points),
                       (num_points, 1, 1, 1))
  lin_batch = x + epsilon * eta
  logits = model.get_logits(lin_batch)
  with sess.as_default():
    log_prob_adv_array = sess.run(logits,
                                  feed_dict={x: x_data})
  return log_prob_adv_array

def linear_extrapolation_plot(log_prob_adv_array, y, file_name,
                              min_epsilon=-10, max_epsilon=10,
                              num_points=21):
  """Generate linear extrapolation plot.

  Args:
      log_prob_adv_array: Numpy array containing log probabilities
      y: Tf placeholder for the labels
      file_name: Plot filename
      min_epsilon: Minimum value of epsilon over the interval
      max_epsilon: Maximum value of epsilon over the interval
      num_points: Number of points used to interpolate
  """
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  figure = plt.figure()
  figure.canvas.set_window_title('Cleverhans: Linear Extrapolation Plot')

  correct_idx = np.argmax(y, axis=0)
  fig = plt.figure()
  plt.xlabel('Epsilon')
  plt.ylabel('Logits')
  x_axis = np.linspace(min_epsilon, max_epsilon, num_points)
  plt.xlim(min_epsilon - 1, max_epsilon + 1)
  for i in range(y.shape[0]):
    if i == correct_idx:
      ls = '-'
      linewidth = 5
    else:
      ls = '--'
      linewidth = 2
    plt.plot(
        x_axis,
        log_prob_adv_array[:, i],
        ls=ls,
        linewidth=linewidth,
        label='{}'.format(i))
  plt.legend(loc='best', fontsize=14)
  plt.show()
  fig.savefig(file_name)
  plt.clf()
  return figure
