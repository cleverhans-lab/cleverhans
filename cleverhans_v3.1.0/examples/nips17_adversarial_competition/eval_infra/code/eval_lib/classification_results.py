"""Module with classes to compute, read and store classification results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from io import BytesIO
from io import StringIO
import logging
import os
import pickle
import time
from six import iteritems
from six import iterkeys
from six import itervalues
from six import PY3

KIND_CLASSIFICATION_BATCH = u'ClassificationBatch'
CLASSIFICATION_BATCH_ID_PATTERN = u'CBATCH{:06}'

CLASSIFICATION_BATCHES_SUBDIR = 'classification_batches'

TO_STR_MAX_BATCHES = 10

MAX_ALLOWED_CLASSIFICATION_RESULT_SIZE = 10000


def read_classification_results(storage_client, file_path):
  """Reads classification results from the file in Cloud Storage.

  This method reads file with classification results produced by running
  defense on singe batch of adversarial images.

  Args:
    storage_client: instance of CompetitionStorageClient or None for local file
    file_path: path of the file with results

  Returns:
    dictionary where keys are image names or IDs and values are classification
      labels
  """
  if storage_client:
    # file on Cloud
    success = False
    retry_count = 0
    while retry_count < 4:
      try:
        blob = storage_client.get_blob(file_path)
        if not blob:
          return {}
        if blob.size > MAX_ALLOWED_CLASSIFICATION_RESULT_SIZE:
          logging.warning('Skipping classification result because it''s too '
                          'big: %d bytes for %s', blob.size, file_path)
          return None
        buf = BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        success = True
        break
      except Exception:
        retry_count += 1
        time.sleep(5)
    if not success:
      return None
  else:
    # local file
    try:
      with open(file_path, 'rb') as f:
        buf = BytesIO(f.read())
    except IOError:
      return None
  result = {}
  if PY3:
    buf = StringIO(buf.read().decode('UTF-8'))
  for row in csv.reader(buf):
    try:
      image_filename = row[0]
      if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
        image_filename = image_filename[:image_filename.rfind('.')]
      label = int(row[1])
    except (IndexError, ValueError):
      continue
    result[image_filename] = label
  return result


def analyze_one_classification_result(storage_client, file_path,
                                      adv_batch, dataset_batches,
                                      dataset_meta):
  """Reads and analyzes one classification result.

  This method reads file with classification result and counts
  how many images were classified correctly and incorrectly,
  how many times target class was hit and total number of images.

  Args:
    storage_client: instance of CompetitionStorageClient
    file_path: result file path
    adv_batch: AversarialBatches.data[adv_batch_id]
      adv_batch_id is stored in each ClassificationBatch entity
    dataset_batches: instance of DatasetBatches
    dataset_meta: instance of DatasetMetadata

  Returns:
    Tuple of (count_correctly_classified, count_errors, count_hit_target_class,
    num_images)
  """
  class_result = read_classification_results(storage_client, file_path)
  if class_result is None:
    return 0, 0, 0, 0
  adv_images = adv_batch['images']
  dataset_batch_images = (
      dataset_batches.data[adv_batch['dataset_batch_id']]['images'])
  count_correctly_classified = 0
  count_errors = 0
  count_hit_target_class = 0
  num_images = 0
  for adv_img_id, label in iteritems(class_result):
    if adv_img_id not in adv_images:
      continue
    num_images += 1
    clean_image_id = adv_images[adv_img_id]['clean_image_id']
    dataset_image_id = (
        dataset_batch_images[clean_image_id]['dataset_image_id'])
    if label == dataset_meta.get_true_label(dataset_image_id):
      count_correctly_classified += 1
    else:
      count_errors += 1
    if label == dataset_meta.get_target_class(dataset_image_id):
      count_hit_target_class += 1
  return (count_correctly_classified, count_errors,
          count_hit_target_class, num_images)


class ResultMatrix(object):
  """Sparse matrix where rows and columns are indexed using string.

  This matrix is used to store resutls of the competition evaluation.
  """

  def __init__(self, default_value=0):
    """Initializes empty matrix."""
    self._items = {}
    self._dim0 = set()
    self._dim1 = set()
    self._default_value = default_value

  @property
  def dim0(self):
    """Returns set of rows."""
    return self._dim0

  @property
  def dim1(self):
    """Returns set of columns."""
    return self._dim1

  def __getitem__(self, key):
    """Returns element of the matrix indexed by given key.

    Args:
      key: tuple of (row_idx, column_idx)

    Returns:
      Element of the matrix

    Raises:
      IndexError: if key is invalid.
    """
    if not isinstance(key, tuple) or len(key) != 2:
      raise IndexError('Invalid index: {0}'.format(key))
    return self._items.get(key, self._default_value)

  def __setitem__(self, key, value):
    """Sets element of the matrix at position indexed by key.

    Args:
      key: tuple of (row_idx, column_idx)
      value: new value of the element of the matrix

    Raises:
      IndexError: if key is invalid.
    """
    if not isinstance(key, tuple) or len(key) != 2:
      raise IndexError('Invalid index: {0}'.format(key))
    self._dim0.add(key[0])
    self._dim1.add(key[1])
    self._items[key] = value

  def save_to_file(self, filename, remap_dim0=None, remap_dim1=None):
    """Saves matrix to the file.

    Args:
      filename: name of the file where to save matrix
      remap_dim0: dictionary with mapping row indices to row names which should
        be saved to file. If none then indices will be used as names.
      remap_dim1: dictionary with mapping column indices to column names which
        should be saved to file. If none then indices will be used as names.
    """
    # rows - first index
    # columns - second index
    with open(filename, 'w') as fobj:
      columns = list(sorted(self._dim1))
      for col in columns:
        fobj.write(',')
        fobj.write(str(remap_dim1[col] if remap_dim1 else col))
      fobj.write('\n')
      for row in sorted(self._dim0):
        fobj.write(str(remap_dim0[row] if remap_dim0 else row))
        for col in columns:
          fobj.write(',')
          fobj.write(str(self[row, col]))
        fobj.write('\n')


class ClassificationBatches(object):
  """Class which generates and stores classification batches.

  Each classification batch contains result of the classification of one
  batch of adversarial images.
  """

  def __init__(self, datastore_client, storage_client, round_name):
    """Initializes ClassificationBatches.

    Args:
      datastore_client: instance of CompetitionDatastoreClient
      storage_client: instance of CompetitionStorageClient
      round_name: name of the round
    """
    self._datastore_client = datastore_client
    self._storage_client = storage_client
    self._round_name = round_name
    # Data is dict of dicts {CLASSIFICATION_BATCH_ID: { ... }}
    self._data = {}

  def serialize(self, fobj):
    """Serializes data stored in this class."""
    pickle.dump(self._data, fobj)

  def deserialize(self, fobj):
    """Deserializes data from file into this class."""
    self._data = pickle.load(fobj)

  @property
  def data(self):
    """Returns dictionary with data."""
    return self._data

  def __getitem__(self, key):
    """Returns one classification batch by given key."""
    return self._data[key]

  def init_from_adversarial_batches_write_to_datastore(self, submissions,
                                                       adv_batches):
    """Populates data from adversarial batches and writes to datastore.

    Args:
      submissions: instance of CompetitionSubmissions
      adv_batches: instance of AversarialBatches
    """
    # prepare classification batches
    idx = 0
    for s_id in iterkeys(submissions.defenses):
      for adv_id in iterkeys(adv_batches.data):
        class_batch_id = CLASSIFICATION_BATCH_ID_PATTERN.format(idx)
        idx += 1
        self.data[class_batch_id] = {
            'adversarial_batch_id': adv_id,
            'submission_id': s_id,
            'result_path': os.path.join(
                self._round_name,
                CLASSIFICATION_BATCHES_SUBDIR,
                s_id + '_' + adv_id + '.csv')
        }
    # save them to datastore
    client = self._datastore_client
    with client.no_transact_batch() as batch:
      for key, value in iteritems(self.data):
        entity = client.entity(client.key(KIND_CLASSIFICATION_BATCH, key))
        entity.update(value)
        batch.put(entity)

  def init_from_datastore(self):
    """Initializes data by reading it from the datastore."""
    self._data = {}
    client = self._datastore_client
    for entity in client.query_fetch(kind=KIND_CLASSIFICATION_BATCH):
      class_batch_id = entity.key.flat_path[-1]
      self.data[class_batch_id] = dict(entity)

  def read_batch_from_datastore(self, class_batch_id):
    """Reads and returns single batch from the datastore."""
    client = self._datastore_client
    key = client.key(KIND_CLASSIFICATION_BATCH, class_batch_id)
    result = client.get(key)
    if result is not None:
      return dict(result)
    else:
      raise KeyError(
          'Key {0} not found in the datastore'.format(key.flat_path))

  def compute_classification_results(self, adv_batches, dataset_batches,
                                     dataset_meta, defense_work=None):
    """Computes classification results.

    Args:
      adv_batches: instance of AversarialBatches
      dataset_batches: instance of DatasetBatches
      dataset_meta: instance of DatasetMetadata
      defense_work: instance of DefenseWorkPieces

    Returns:
      accuracy_matrix, error_matrix, hit_target_class_matrix,
      processed_images_count
    """
    class_batch_to_work = {}
    if defense_work:
      for v in itervalues(defense_work.work):
        class_batch_to_work[v['output_classification_batch_id']] = v

    # accuracy_matrix[defense_id, attack_id] = num correctly classified
    accuracy_matrix = ResultMatrix()
    # error_matrix[defense_id, attack_id] = num misclassfied
    error_matrix = ResultMatrix()
    # hit_target_class_matrix[defense_id, attack_id] = num hit target class
    hit_target_class_matrix = ResultMatrix()
    # processed_images_count[defense_id] = num processed images by defense
    processed_images_count = {}

    total_count = len(self.data)
    processed_count = 0
    logging.info('Processing %d files with classification results',
                 len(self.data))
    for k, v in iteritems(self.data):
      if processed_count % 100 == 0:
        logging.info('Processed %d out of %d classification results',
                     processed_count, total_count)
      processed_count += 1
      defense_id = v['submission_id']
      adv_batch = adv_batches.data[v['adversarial_batch_id']]
      attack_id = adv_batch['submission_id']

      work_item = class_batch_to_work.get(k)
      required_work_stats = ['stat_correct', 'stat_error', 'stat_target_class',
                             'stat_num_images']
      if work_item and work_item['error']:
        # ignore batches with error
        continue
      if work_item and all(work_item.get(i) is not None
                           for i in required_work_stats):
        count_correctly_classified = work_item['stat_correct']
        count_errors = work_item['stat_error']
        count_hit_target_class = work_item['stat_target_class']
        num_images = work_item['stat_num_images']
      else:
        logging.warning('Recomputing accuracy for classification batch %s', k)
        (count_correctly_classified, count_errors, count_hit_target_class,
         num_images) = analyze_one_classification_result(
             self._storage_client, v['result_path'], adv_batch, dataset_batches,
             dataset_meta)

      # update accuracy and hit target class
      accuracy_matrix[defense_id, attack_id] += count_correctly_classified
      error_matrix[defense_id, attack_id] += count_errors
      hit_target_class_matrix[defense_id, attack_id] += count_hit_target_class
      # update number of processed images
      processed_images_count[defense_id] = (
          processed_images_count.get(defense_id, 0) + num_images)
    return (accuracy_matrix, error_matrix, hit_target_class_matrix,
            processed_images_count)

  def __str__(self):
    """Returns human readable string representation, useful for debugging."""
    buf = StringIO()
    for idx, (class_batch_id, class_val) in enumerate(iteritems(self.data)):
      if idx >= TO_STR_MAX_BATCHES:
        buf.write(u'  ...\n')
        break
      buf.write(u'  ClassBatch "{0}"\n'.format(class_batch_id))
      buf.write(u'    {0}\n'.format(str(class_val)))
    return buf.getvalue()
