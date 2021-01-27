"""Various helpers for the dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import hashlib
import logging
import os
import shutil
import subprocess

import numpy as np
from PIL import Image

from six import iteritems


class DatasetMetadata(object):
  """Helper class which loads and stores dataset metadata.

  Dataset metadata stored by this class contains true labels and target classes
  for all images in the dataset.
  """

  def __init__(self, fobj):
    """Initializes instance of DatasetMetadata.

    Args:
      fobj: file object
    """
    self._true_labels = {}
    self._target_classes = {}
    reader = csv.reader(fobj)
    header_row = next(reader)
    try:
      row_idx_image_id = header_row.index('ImageId')
      row_idx_true_label = header_row.index('TrueLabel')
      row_idx_target_class = header_row.index('TargetClass')
    except ValueError:
      raise IOError('Invalid format of the dataset metadata.')
    for row in reader:
      if len(row) < len(header_row):
        # skip partial or empty lines
        continue
      try:
        image_id = row[row_idx_image_id]
        self._true_labels[image_id] = int(row[row_idx_true_label])
        self._target_classes[image_id] = int(row[row_idx_target_class])
      except (IndexError, ValueError):
        raise IOError('Invalid format of dataset metadata')

  def get_true_label(self, image_id):
    """Returns true label for image with given ID."""
    return self._true_labels[image_id]

  def get_target_class(self, image_id):
    """Returns target class for image with given ID."""
    return self._target_classes[image_id]

  def save_target_classes_for_batch(self,
                                    filename,
                                    image_batches,
                                    batch_id):
    """Saves file with target class for given dataset batch.

    Args:
      filename: output filename
      image_batches: instance of ImageBatchesBase with dataset batches
      batch_id: dataset batch ID
    """
    images = image_batches.data[batch_id]['images']
    with open(filename, 'w') as f:
      for image_id, image_val in iteritems(images):
        target_class = self.get_target_class(image_val['dataset_image_id'])
        f.write('{0}.png,{1}\n'.format(image_id, target_class))


def enforce_epsilon_and_compute_hash(dataset_batch_dir, adv_dir, output_dir,
                                     epsilon):
  """Enforces size of perturbation on images, and compute hashes for all images.

  Args:
    dataset_batch_dir: directory with the images of specific dataset batch
    adv_dir: directory with generated adversarial images
    output_dir: directory where to copy result
    epsilon: size of perturbation

  Returns:
    dictionary with mapping form image ID to hash.
  """
  dataset_images = [f for f in os.listdir(dataset_batch_dir)
                    if f.endswith('.png')]
  image_hashes = {}
  resize_warning = False
  for img_name in dataset_images:
    if not os.path.exists(os.path.join(adv_dir, img_name)):
      logging.warning('Image %s not found in the output', img_name)
      continue
    image = np.array(
        Image.open(os.path.join(dataset_batch_dir, img_name)).convert('RGB'))
    image = image.astype('int32')
    image_max_clip = np.clip(image + epsilon, 0, 255).astype('uint8')
    image_min_clip = np.clip(image - epsilon, 0, 255).astype('uint8')
    # load and resize adversarial image if needed
    adv_image = Image.open(os.path.join(adv_dir, img_name)).convert('RGB')
    # Image.size is reversed compared to np.array.shape
    if adv_image.size[::-1] != image.shape[:2]:
      resize_warning = True
      adv_image = adv_image.resize((image.shape[1], image.shape[0]),
                                   Image.BICUBIC)
    adv_image = np.array(adv_image)
    clipped_adv_image = np.clip(adv_image,
                                image_min_clip,
                                image_max_clip)
    Image.fromarray(clipped_adv_image).save(os.path.join(output_dir, img_name))
    # compute hash
    image_hashes[img_name[:-4]] = hashlib.sha1(
        clipped_adv_image.view(np.uint8)).hexdigest()
  if resize_warning:
    logging.warning('One or more adversarial images had incorrect size')
  return image_hashes


def download_dataset(storage_client, image_batches, target_dir,
                     local_dataset_copy=None):
  """Downloads dataset, organize it by batches and rename images.

  Args:
    storage_client: instance of the CompetitionStorageClient
    image_batches: subclass of ImageBatchesBase with data about images
    target_dir: target directory, should exist and be empty
    local_dataset_copy: directory with local dataset copy, if local copy is
      available then images will be takes from there instead of Cloud Storage

  Data in the target directory will be organized into subdirectories by batches,
  thus path to each image will be "target_dir/BATCH_ID/IMAGE_ID.png"
  where BATCH_ID - ID of the batch (key of image_batches.data),
  IMAGE_ID - ID of the image (key of image_batches.data[batch_id]['images'])
  """
  for batch_id, batch_value in iteritems(image_batches.data):
    batch_dir = os.path.join(target_dir, batch_id)
    os.mkdir(batch_dir)
    for image_id, image_val in iteritems(batch_value['images']):
      dst_filename = os.path.join(batch_dir, image_id + '.png')
      # try to use local copy first
      if local_dataset_copy:
        local_filename = os.path.join(local_dataset_copy,
                                      os.path.basename(image_val['image_path']))
        if os.path.exists(local_filename):
          shutil.copyfile(local_filename, dst_filename)
          continue
      # download image from cloud
      cloud_path = ('gs://' + storage_client.bucket_name
                    + '/' + image_val['image_path'])
      if not os.path.exists(dst_filename):
        subprocess.call(['gsutil', 'cp', cloud_path, dst_filename])
