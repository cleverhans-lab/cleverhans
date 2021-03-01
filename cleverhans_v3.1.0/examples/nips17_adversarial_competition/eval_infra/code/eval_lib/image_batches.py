"""Module with classes to read and store image batches.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from io import BytesIO
from io import StringIO
import itertools
import logging
import os
import zipfile

from six import iteritems
from six import iterkeys
from six import itervalues


# Cloud Datastore constants
KIND_DATASET_BATCH = u'DatasetBatch'
KIND_DATASET_IMAGE = u'DatasetImage'
KIND_ADVERSARIAL_BATCH = u'AdversarialBatch'
KIND_ADVERSARIAL_IMAGE = u'AdversarialImage'

# Cloud Datastore ID patterns
DATASET_BATCH_ID_PATTERN = u'BATCH{:03}'
DATASET_IMAGE_ID_PATTERN = u'IMG{:06}'
ADVERSARIAL_BATCH_ID_PATTERN = u'ADVBATCH{:03}'
ADVERSARIAL_IMAGE_ID_PATTERN = u'ADV{:06}'

# Default list of possible adversarial peturbations
DEFAULT_EPSILON = [4, 8, 12, 16]

# Constants for __str__
TO_STR_MAX_BATCHES = 5
TO_STR_MAX_IMAGES_PER_BATCH = 10


class ImageBatchesBase(object):
  """Base class to store image batches.

  Subclasses of this class are used to store batches of images from the dataset
  or batches of adversarial images.
  """

  def __init__(self, datastore_client, entity_kind_batches, entity_kind_images):
    """Initialize ImageBatchesBase.

    Args:
      datastore_client: instance of the CompetitionDatastoreClient
      entity_kind_batches: Cloud Datastore entity kind which is used to store
        batches of images.
      entity_kind_images: Cloud Datastore entity kind which is used to store
        individual images.
    """
    self._datastore_client = datastore_client
    self._entity_kind_batches = entity_kind_batches
    self._entity_kind_images = entity_kind_images
    # data is a dictionary with following structure:
    #  self._data[batch_id] = {
    #    'batch_k1': batch_v1,
    #    ...
    #    'batch_kN': batch_vN,
    #    'images': {
    #      image_id: { 'img_k1': img_v1, ... }
    #    }
    #  }
    self._data = {}

  def _write_single_batch_images_internal(self, batch_id, client_batch):
    """Helper method to write images from single batch into datastore."""
    client = self._datastore_client
    batch_key = client.key(self._entity_kind_batches, batch_id)
    for img_id, img in iteritems(self._data[batch_id]['images']):
      img_entity = client.entity(
          client.key(self._entity_kind_images, img_id, parent=batch_key))
      for k, v in iteritems(img):
        img_entity[k] = v
      client_batch.put(img_entity)

  def write_to_datastore(self):
    """Writes all image batches to the datastore."""
    client = self._datastore_client
    with client.no_transact_batch() as client_batch:
      for batch_id, batch_data in iteritems(self._data):
        batch_key = client.key(self._entity_kind_batches, batch_id)
        batch_entity = client.entity(batch_key)
        for k, v in iteritems(batch_data):
          if k != 'images':
            batch_entity[k] = v
        client_batch.put(batch_entity)
        self._write_single_batch_images_internal(batch_id, client_batch)

  def write_single_batch_images_to_datastore(self, batch_id):
    """Writes only images from one batch to the datastore."""
    client = self._datastore_client
    with client.no_transact_batch() as client_batch:
      self._write_single_batch_images_internal(batch_id, client_batch)

  def init_from_datastore(self):
    """Initializes batches by reading from the datastore."""
    self._data = {}
    for entity in self._datastore_client.query_fetch(
        kind=self._entity_kind_batches):
      batch_id = entity.key.flat_path[-1]
      self._data[batch_id] = dict(entity)
      self._data[batch_id]['images'] = {}
    for entity in self._datastore_client.query_fetch(
        kind=self._entity_kind_images):
      batch_id = entity.key.flat_path[-3]
      image_id = entity.key.flat_path[-1]
      self._data[batch_id]['images'][image_id] = dict(entity)

  @property
  def data(self):
    """Dictionary with data."""
    return self._data

  def __getitem__(self, key):
    """Returns specific batch by its key."""
    return self._data[key]

  def add_batch(self, batch_id, batch_properties=None):
    """Adds batch with give ID and list of properties."""
    if batch_properties is None:
      batch_properties = {}
    if not isinstance(batch_properties, dict):
      raise ValueError('batch_properties has to be dict, however it was: '
                       + str(type(batch_properties)))
    self._data[batch_id] = batch_properties.copy()
    self._data[batch_id]['images'] = {}

  def add_image(self, batch_id, image_id, image_properties=None):
    """Adds image to given batch."""
    if batch_id not in self._data:
      raise KeyError('Batch with ID "{0}" does not exist'.format(batch_id))
    if image_properties is None:
      image_properties = {}
    if not isinstance(image_properties, dict):
      raise ValueError('image_properties has to be dict, however it was: '
                       + str(type(image_properties)))
    self._data[batch_id]['images'][image_id] = image_properties.copy()

  def count_num_images(self):
    """Counts total number of images in all batches."""
    return sum([len(v['images']) for v in itervalues(self.data)])

  def __str__(self):
    """Returns human readable representation, which is useful for debugging."""
    buf = StringIO()
    for batch_idx, (batch_id, batch_val) in enumerate(iteritems(self.data)):
      if batch_idx >= TO_STR_MAX_BATCHES:
        buf.write(u'...\n')
        break
      buf.write(u'BATCH "{0}"\n'.format(batch_id))
      for k, v in iteritems(batch_val):
        if k != 'images':
          buf.write(u'  {0}: {1}\n'.format(k, v))
      for img_idx, img_id in enumerate(iterkeys(batch_val['images'])):
        if img_idx >= TO_STR_MAX_IMAGES_PER_BATCH:
          buf.write(u'  ...')
          break
        buf.write(u'  IMAGE "{0}" -- {1}\n'.format(img_id,
                                                   batch_val['images'][img_id]))
      buf.write(u'\n')
    return buf.getvalue()


class DatasetBatches(ImageBatchesBase):
  """Class which stores batches of images from the dataset."""

  def __init__(self, datastore_client, storage_client, dataset_name):
    """Initializes DatasetBatches.

    Args:
      datastore_client: instance of CompetitionDatastoreClient
      storage_client: instance of CompetitionStorageClient
      dataset_name: name of the dataset ('dev' or 'final')
    """
    super(DatasetBatches, self).__init__(
        datastore_client=datastore_client,
        entity_kind_batches=KIND_DATASET_BATCH,
        entity_kind_images=KIND_DATASET_IMAGE)
    self._storage_client = storage_client
    self._dataset_name = dataset_name

  def _read_image_list(self, skip_image_ids=None):
    """Reads list of dataset images from the datastore."""
    if skip_image_ids is None:
      skip_image_ids = []
    images = self._storage_client.list_blobs(
        prefix=os.path.join('dataset', self._dataset_name) + '/')
    zip_files = [i for i in images if i.endswith('.zip')]
    if len(zip_files) == 1:
      # we have a zip archive with images
      zip_name = zip_files[0]
      logging.info('Reading list of images from zip file %s', zip_name)
      blob = self._storage_client.get_blob(zip_name)
      buf = BytesIO()
      logging.info('Downloading zip')
      blob.download_to_file(buf)
      buf.seek(0)
      logging.info('Reading content of the zip')
      with zipfile.ZipFile(buf) as f:
        images = [os.path.join(zip_name, os.path.basename(n))
                  for n in f.namelist() if n.endswith('.png')]
      buf.close()
      logging.info('Found %d images', len(images))
    else:
      # we have just a directory with images, filter non-PNG files
      logging.info('Reading list of images from png files in storage')
      images = [i for i in images if i.endswith('.png')]
      logging.info('Found %d images', len(images))
    # filter images which should be skipped
    images = [i for i in images
              if os.path.basename(i)[:-4] not in skip_image_ids]
    # assign IDs to images
    images = [(DATASET_IMAGE_ID_PATTERN.format(idx), i)
              for idx, i in enumerate(sorted(images))]
    return images

  def init_from_storage_write_to_datastore(self,
                                           batch_size=100,
                                           allowed_epsilon=None,
                                           skip_image_ids=None,
                                           max_num_images=None):
    """Initializes dataset batches from the list of images in the datastore.

    Args:
      batch_size: batch size
      allowed_epsilon: list of allowed epsilon or None to use default
      skip_image_ids: list of image ids to skip
      max_num_images: maximum number of images to read
    """
    if allowed_epsilon is None:
      allowed_epsilon = copy.copy(DEFAULT_EPSILON)
    # init dataset batches from data in storage
    self._dataset_batches = {}
    # read all blob names from storage
    images = self._read_image_list(skip_image_ids)
    if max_num_images:
      images = images[:max_num_images]
    for batch_idx, batch_start in enumerate(range(0, len(images), batch_size)):
      batch = images[batch_start:batch_start+batch_size]
      batch_id = DATASET_BATCH_ID_PATTERN.format(batch_idx)
      batch_epsilon = allowed_epsilon[batch_idx % len(allowed_epsilon)]
      self.add_batch(batch_id, {'epsilon': batch_epsilon})
      for image_id, image_path in batch:
        self.add_image(batch_id, image_id,
                       {'dataset_image_id': os.path.basename(image_path)[:-4],
                        'image_path': image_path})
    # write data to datastore
    self.write_to_datastore()


class AversarialBatches(ImageBatchesBase):
  """Class which stores batches of adversarial images generated by attacks."""

  def __init__(self, datastore_client):
    """Initializes AversarialBatches.

    Args:
      datastore_client: instance of CompetitionDatastoreClient
    """
    super(AversarialBatches, self).__init__(
        datastore_client=datastore_client,
        entity_kind_batches=KIND_ADVERSARIAL_BATCH,
        entity_kind_images=KIND_ADVERSARIAL_IMAGE)

  def init_from_dataset_and_submissions_write_to_datastore(
      self, dataset_batches, attack_submission_ids):
    """Init list of adversarial batches from dataset batches and submissions.

    Args:
      dataset_batches: instances of DatasetBatches
      attack_submission_ids: iterable with IDs of all (targeted and nontargeted)
        attack submissions, could be obtains as
        CompetitionSubmissions.get_all_attack_ids()
    """
    batches_x_attacks = itertools.product(dataset_batches.data.keys(),
                                          attack_submission_ids)
    for idx, (dataset_batch_id, attack_id) in enumerate(batches_x_attacks):
      adv_batch_id = ADVERSARIAL_BATCH_ID_PATTERN.format(idx)
      self.add_batch(adv_batch_id,
                     {'dataset_batch_id': dataset_batch_id,
                      'submission_id': attack_id})
    self.write_to_datastore()

  def count_generated_adv_examples(self):
    """Returns total number of all generated adversarial examples."""
    result = {}
    for v in itervalues(self.data):
      s_id = v['submission_id']
      result[s_id] = result.get(s_id, 0) + len(v['images'])
    return result
