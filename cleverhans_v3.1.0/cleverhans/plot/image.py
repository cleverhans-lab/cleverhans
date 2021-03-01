"""
Functionality for displaying or saving images.
"""
from tempfile import mkstemp
import os
import platform

import numpy as np
from PIL import Image

from cleverhans.utils import shell_call

def show(ndarray, min_val=None, max_val=None):
  """
  Display an image.
  :param ndarray: The image as an ndarray
  :param min_val: The minimum pixel value in the image format
  :param max_val: The maximum pixel valie in the image format
    If min_val and max_val are not specified, attempts to
    infer whether the image is in any of the common ranges:
      [0, 1], [-1, 1], [0, 255]
    This can be ambiguous, so it is better to specify if known.
  """

  # Create a temporary file with the suffix '.png'.
  fd, path = mkstemp(suffix='.png')
  os.close(fd)
  save(path, ndarray, min_val, max_val)
  shell_call(VIEWER_COMMAND + [path])



def save(path, ndarray, min_val=None, max_val=None):
  """
  Save an image, represented as an ndarray, to the filesystem
  :param path: string, filepath
  :param ndarray: The image as an ndarray
  :param min_val: The minimum pixel value in the image format
  :param max_val: The maximum pixel valie in the image format
    If min_val and max_val are not specified, attempts to
    infer whether the image is in any of the common ranges:
      [0, 1], [-1, 1], [0, 255]
    This can be ambiguous, so it is better to specify if known.
  """
  as_pil(ndarray, min_val, max_val).save(path)

def as_pil(ndarray, min_val=None, max_val=None):
  """
  Converts an ndarray to a PIL image.
  :param ndarray: The numpy ndarray to convert
  :param min_val: The minimum pixel value in the image format
  :param max_val: The maximum pixel valie in the image format
    If min_val and max_val are not specified, attempts to
    infer whether the image is in any of the common ranges:
      [0, 1], [-1, 1], [0, 255]
    This can be ambiguous, so it is better to specify if known.
  """

  assert isinstance(ndarray, np.ndarray)

  # rows x cols for grayscale image
  # rows x cols x channels for color
  assert ndarray.ndim in [2, 3]
  if ndarray.ndim == 3:
    channels = ndarray.shape[2]
    # grayscale or RGB
    assert channels in [1, 3]

  actual_min = ndarray.min()
  actual_max = ndarray.max()

  if min_val is not None:
    assert actual_min >= min_val
    assert actual_max <= max_val

  if np.issubdtype(ndarray.dtype, np.floating):
    if min_val is None:
      if actual_min < -1.:
        raise ValueError("Unrecognized range")
      if actual_min < 0:
        min_val = -1.
      else:
        min_val = 0.
    if max_val is None:
      if actual_max > 255.:
        raise ValueError("Unrecognized range")
      if actual_max > 1.:
        max_val = 255.
      else:
        max_val = 1.
    ndarray = (ndarray - min_val)
    value_range = max_val - min_val
    ndarray *= (255. / value_range)
    ndarray = np.cast['uint8'](ndarray)
  elif 'int' in str(ndarray.dtype):
    if min_val is not None:
      assert min_val == 0
    else:
      assert actual_min >= 0.
    if max_val is not None:
      assert max_val == 255
    else:
      assert actual_max <= 255.
  else:
    raise ValueError("Unrecognized dtype")

  out = Image.fromarray(ndarray)

  return out

def make_grid(image_batch):
  """
  Turns a batch of images into one big image.
  :param image_batch: ndarray, shape (batch_size, rows, cols, channels)
  :returns : a big image containing all `batch_size` images in a grid
  """
  m, ir, ic, ch = image_batch.shape

  pad = 3

  padded = np.zeros((m, ir + pad * 2, ic + pad * 2, ch))
  padded[:, pad:-pad, pad:-pad, :] = image_batch

  m, ir, ic, ch = padded.shape

  pr = int(np.sqrt(m))
  pc = int(np.ceil(float(m) / pr))
  extra_m = pr * pc
  assert extra_m > m

  padded = np.concatenate((padded, np.zeros((extra_m - m, ir, ic, ch))), axis=0)

  row_content = np.split(padded, pr)
  row_content = [np.split(content, pc) for content in row_content]
  rows = [np.concatenate(content, axis=2) for content in row_content]
  grid = np.concatenate(rows, axis=1)
  assert grid.shape[0] == 1, grid.shape
  grid = grid[0]

  return grid


if platform.system() == 'Darwin':
  VIEWER_COMMAND = ['open', '-a', 'Preview']
else:
  VIEWER_COMMAND = ['eog', '--new-instance']
