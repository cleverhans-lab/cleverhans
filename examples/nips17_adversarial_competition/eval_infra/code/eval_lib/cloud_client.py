"""Helper classes and wrappers to access Google Cloud.

Google Cloud API is encapsulated with these wrappers, so it's easier to
test the code with help of fake (declared in testing/fake_cloud_client.py).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import random
import time
import traceback as tb

from google.cloud import datastore
from google.cloud import storage
from google.cloud.exceptions import TooManyRequests

# To resolve InsecurePlatformWarning
try:
  import urllib3.contrib.pyopenssl
  urllib3.contrib.pyopenssl.inject_into_urllib3()
  print('Pyopenssl fix for urllib3 succesfully injected.')
except ImportError:
  print('Failed to inject pyopenssl fix for urllib3.')

# Cloud Datastore has 500 mutations per batch limit.
MAX_MUTATIONS_IN_BATCH = 500


class CompetitionStorageClient(object):
  """Client wrapper to access Google Cloud Storage."""

  def __init__(self, project_id, bucket_name):
    """Initialize client with project id and name of the storage bucket."""
    self.project_id = project_id
    self.bucket_name = bucket_name
    self.client = storage.Client(project=project_id)
    self.bucket = self.client.get_bucket(bucket_name)

  def list_blobs(self, prefix=''):
    """Lists names of all blobs by their prefix."""
    return [b.name for b in self.bucket.list_blobs(prefix=prefix)]

  def get_blob(self, blob_name):
    """Gets google.cloud.storage.blob.Blob object by blob name."""
    return self.bucket.get_blob(blob_name)

  def new_blob(self, blob_name):
    """Creates new storage blob with provided name."""
    return storage.Blob(blob_name, self.bucket)


class NoTransactionBatch(object):
  """No transaction batch to write large number of entities.

  Usage:
    client = ...  # instance of CompetitionDatastoreClient
    with NoTransactionBatch(client) as batch:
      batch.put(entity1)
      ...
      batch.put(entityN)
      batch.delete(del_entity1)
      ...
      batch.delete(del_entityM)

  It could be also used via CompetitionDatastoreClient.no_transact_batch:
    client = ...  # instance of CompetitionDatastoreClient
    with client.no_transact_batch() as batch:
      batch.put(entity1)
      ...
      batch.put(entityN)
      batch.delete(del_entity1)
      ...
      batch.delete(del_entityM)

  Most methods of this class are provided to simulate
  google.cloud.datastore.batch.Batch interface, so they could be used
  interchangeably.
  Practically speaking, this class works by maintaining a buffer of
  pending mutations and committing them as soon as the length of the buffer
  reaches MAX_MUTATIONS_IN_BATCH.
  """

  def __init__(self, client):
    """Init NoTransactionBatch with provided CompetitionDatastoreClient."""
    self._client = client
    self._cur_batch = None
    self._num_mutations = 0

  def begin(self):
    """Begins a batch."""
    if self._cur_batch:
      raise ValueError('Previous batch is not committed.')
    self._cur_batch = self._client.batch()
    self._cur_batch.begin()
    self._num_mutations = 0

  def commit(self):
    """Commits all pending mutations."""
    self._cur_batch.commit()
    self._cur_batch = None
    self._num_mutations = 0

  def rollback(self):
    """Rolls back pending mutations.

    Keep in mind that NoTransactionBatch splits all mutations into smaller
    batches and commit them as soon as mutation buffer reaches maximum length.
    That's why rollback method will only roll back pending mutations from the
    buffer, but won't be able to rollback already committed mutations.
    """
    try:
      if self._cur_batch:
        self._cur_batch.rollback()
    except ValueError:
      # ignore "Batch must be in progress to rollback" error
      pass
    self._cur_batch = None
    self._num_mutations = 0

  def put(self, entity):
    """Adds mutation of the entity to the mutation buffer.

    If mutation buffer reaches its capacity then this method commit all pending
    mutations from the buffer and emties it.

    Args:
      entity: entity which should be put into the datastore
    """
    self._cur_batch.put(entity)
    self._num_mutations += 1
    if self._num_mutations >= MAX_MUTATIONS_IN_BATCH:
      self.commit()
      self.begin()

  def delete(self, key):
    """Adds deletion of the entity with given key to the mutation buffer.

    If mutation buffer reaches its capacity then this method commit all pending
    mutations from the buffer and emties it.

    Args:
      key: key of the entity which should be deleted
    """
    self._cur_batch.delete(key)
    self._num_mutations += 1
    if self._num_mutations >= MAX_MUTATIONS_IN_BATCH:
      self.commit()
      self.begin()

  def __enter__(self):
    self.begin()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_type is None:
      self.commit()
    else:
      err = tb.format_exception(exc_type, exc_value, traceback)
      logging.error('Exception occurred during write:\n%s', err)
      self.rollback()


def iterate_with_exp_backoff(base_iter,
                             max_num_tries=6,
                             max_backoff=300.0,
                             start_backoff=4.0,
                             backoff_multiplier=2.0,
                             frac_random_backoff=0.25):
  """Iterate with exponential backoff on failures.

  Useful to wrap results of datastore Query.fetch to avoid 429 error.

  Args:
    base_iter: basic iterator of generator object
    max_num_tries: maximum number of tries for each request
    max_backoff: maximum backoff, in seconds
    start_backoff: initial value of backoff
    backoff_multiplier: backoff multiplier
    frac_random_backoff: fraction of the value of random part of the backoff

  Yields:
    values of yielded by base iterator
  """
  try_number = 0
  if hasattr(base_iter, '__iter__'):
    base_iter = iter(base_iter)
  while True:
    try:
      yield next(base_iter)
      try_number = 0
    except StopIteration:
      break
    except TooManyRequests as e:
      logging.warning('TooManyRequests error: %s', tb.format_exc())
      if try_number >= max_num_tries:
        logging.error('Number of tries exceeded, too many requests: %s', e)
        raise
      # compute sleep time for truncated exponential backoff
      sleep_time = start_backoff * math.pow(backoff_multiplier, try_number)
      sleep_time *= (1.0 + frac_random_backoff * random.random())
      sleep_time = min(sleep_time, max_backoff)
      logging.warning('Too many requests error, '
                      'retrying with exponential backoff %.3f', sleep_time)
      time.sleep(sleep_time)
      try_number += 1


class CompetitionDatastoreClient(object):
  """Client wrapper to access Google Cloud Datastore."""

  def __init__(self, project_id, namespace=None):
    """Init this method with given project id and optional namespace."""
    self._client = datastore.Client(project=project_id, namespace=namespace)

  def key(self, *args, **kwargs):
    """Creates datastore key."""
    return self._client.key(*args, **kwargs)

  def entity(self, key):
    """Creates datastore entity."""
    return datastore.Entity(key)

  def no_transact_batch(self):
    """Starts batch of mutation which is committed without transaction."""
    return NoTransactionBatch(self._client)

  def batch(self):
    """Starts batch of mutations."""
    return self._client.batch()

  def transaction(self):
    """Starts transaction."""
    return self._client.transaction()

  def get(self, key, transaction=None):
    """Retrieves an entity given its key."""
    return self._client.get(key, transaction=transaction)

  def query_fetch(self, **kwargs):
    """Queries datastore (using exponential backoff)."""
    return iterate_with_exp_backoff(self._client.query(**kwargs).fetch())
