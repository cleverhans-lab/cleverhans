"""Library with fake Google Cloud client, used for testing of eval_lib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from io import StringIO
import six


class FakeBlob(object):
  """Fake for google.cloud.storage.blob.Blob to be used in tests."""

  def __init__(self, content):
    """Initializes FakeBlob with given content."""
    if six.PY3 and isinstance(content, str):
      self._content = content.encode()
    else:
      self._content = content
    self.size = len(content)

  def download_to_file(self, fobj):
    """Writes content of this blob into given file object."""
    fobj.write(self._content)


class FakeStorageClient(object):
  """Fake for CompetitionStorageClient to be used in tests."""

  def __init__(self, blobs=None):
    """Inits FakeStorageClient with given blobs.

    Args:
      blobs: either list of blob names or dict with mapping from blob names to
        their content

    Raises:
      TypeError: if blobs argument has invalid type
    """
    if blobs is not None:
      if isinstance(blobs, dict):
        self._blobs = copy.deepcopy(blobs)
      elif isinstance(blobs, list):
        self._blobs = {k: '' for k in blobs}
      else:
        raise TypeError('Invalid type of blobs argument')
    else:
      self._blobs = {}

  def list_blobs(self, prefix=''):
    """Lists names of all blobs by their prefix."""
    return [b for b in self._blobs.keys() if b.startswith(prefix)]

  def get_blob(self, blob_name):
    """Gets google.cloud.storage.blob.Blob object by blob name."""
    if blob_name in self._blobs:
      return FakeBlob(self._blobs[blob_name])
    else:
      return None

  def new_blob(self, blob_name):
    """Creates new storage blob with provided name."""
    del blob_name
    raise NotImplementedError('new_blob is not implemented in fake client.')


class FakeDatastoreKey(object):
  """Fake datastore key.

  Fake datastore key is represented as a list with flat path.
  """

  def __init__(self, *args, **kwargs):
    if 'parent' not in kwargs:
      self._flat_path = args
    else:
      parent = kwargs['parent']
      if not isinstance(parent, FakeDatastoreKey):
        raise ValueError('Invalid type of parent: ' + str(type(parent)))
      self._flat_path = parent.flat_path + args

  @property
  def flat_path(self):
    return self._flat_path

  def __hash__(self):
    return hash(self._flat_path)

  def __eq__(self, other):
    return (isinstance(other, FakeDatastoreKey)
            and (self.flat_path == other.flat_path))

  def __ne__(self, other):
    return not self.__eq__(other)

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return '<FakeDatastoreKey {0}>'.format(self._flat_path)


class FakeDatastoreEntity(dict):
  """Fake Datstore Entity.

  Fake datastore entity is just a dict, which additionally has key property.
  """

  def __init__(self, key):
    super(FakeDatastoreEntity, self).__init__()
    if not isinstance(key, FakeDatastoreKey):
      raise TypeError('Wrong type of key: ' + str(type(key)))
    self._key = key

  @property
  def key(self):
    return self._key

  def __eq__(self, other):
    if not isinstance(other, FakeDatastoreEntity):
      return False
    return other.key == self.key and (set(self.items()) == set(other.items()))

  def __ne__(self, other):
    return not self.__eq__(other)

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    return '<FakeDatastoreEntity: Key={0} Properties={1}>'.format(
        self.key, super(FakeDatastoreEntity, self).__repr__())


def make_entity(key):
  """Helper method to make FakeDatastoreEntity.

  This method allows to path either tuple or FakeDatastoreKey as a key.

  Args:
    key: entity key, either tuple or FakeDatastoreKey

  Returns:
    Instance of FakeDatastoreEntity
  """
  if isinstance(key, tuple):
    key = FakeDatastoreKey(*key)
  return FakeDatastoreEntity(key)


class FakeDatastoreClientBatch(object):
  """Fake for NoTransactionBatch."""

  def __init__(self, fake_datastore_client):
    """Init FakeDatastoreClientBatch."""
    self._fake_datastore_client = fake_datastore_client
    self._mutations = []

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_type is None:
      for m in self._mutations:
        self._fake_datastore_client.put(m)

  def put(self, entity):
    """Adds entity mutation to the batch."""
    assert isinstance(entity, FakeDatastoreEntity)
    self._mutations.append(copy.deepcopy(entity))


class FakeDatastoreClientTransaction(object):
  """Fake for datastore transaction.

  See https://cloud.google.com/datastore/docs/concepts/transactions
  for details of how transactions work in Cloud Datastore.
  """

  def __init__(self, fake_datastore_client):
    """Init FakeDatastoreClientTransaction."""
    self._client = fake_datastore_client
    # snapshot of the data in the fake datastore
    self._data_snapshot = copy.deepcopy(fake_datastore_client.entities)
    # transaction stated: 'init', 'started', 'committed', 'rolledback'
    self._state = 'init'
    # list of mutations in this transactions in sequential order
    # each mutation is instance of FakeDatastoreEntity
    self._mutations = []
    # set of keys read in this transaction
    self._read_keys = set()

  def _check_transaction_started(self):
    """Helper method to check that transaction has been started."""
    if self._state != 'started':
      raise ValueError(("Invalid state of transaction, "
                        "expected started, was %s") % self._state)

  def _check_update_state(self, old_state, new_state):
    """Checks old state and updates it to new state."""
    if self._state != old_state:
      raise ValueError('Invalid state of transaction, expected %s, was %s' %
                       (old_state, self._state))
    self._state = new_state

  def begin(self):
    """Begins transaction."""
    self._check_update_state('init', 'started')

  def commit(self):
    """Commits transaction."""
    self._check_transaction_started()
    # before committing transaction verity that all entities which
    # were read or updated in the transaction were not modified outside
    # of transaction
    touched_keys = self._read_keys | set([e.key for e in self._mutations])
    for k in touched_keys:
      old_value = self._data_snapshot.get(k)
      cur_value = self._client.entities.get(k)
      if old_value != cur_value:
        self.rollback()
        raise Exception('Transaction can not be committed due to '
                        'conflicted updates in datastore.')
    # commit all changes
    self._state = 'committed'
    for m in self._mutations:
      self._client.put(m)

  def rollback(self):
    """Rolls back current transaction."""
    self._check_update_state('started', 'rolledback')
    self._mutations = []

  def put(self, entity):
    """Puts entity to datastore."""
    assert isinstance(entity, FakeDatastoreEntity)
    self._check_transaction_started()
    self._mutations.append(copy.deepcopy(entity))

  def get(self, key):
    """Gets entity from the datastore."""
    assert isinstance(key, FakeDatastoreKey)
    self._check_transaction_started()
    self._read_keys.add(key)
    return copy.deepcopy(self._data_snapshot.get(key))

  def __enter__(self):
    self.begin()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_type is None:
      self.commit()
    else:
      self.rollback()


_QUERY_FILTER_OPERATOR = {
    '<': lambda x, y: x < y,
    '<=': lambda x, y: x <= y,
    '=': lambda x, y: x == y,
    '>': lambda x, y: x > y,
    '>=': lambda x, y: x >= y,
}


class FakeDatastoreClient(object):
  """Fake for CompetitionDatastoreClient."""

  def __init__(self, entities=None):
    """Init FakeDatastoreClient with specified entities."""
    self._transaction_hook = None
    if isinstance(entities, list):
      self._entities = {e.key: e for e in entities}
    elif isinstance(entities, dict):
      self._entities = entities
    elif entities is None:
      self._entities = {}
    else:
      raise ValueError('Invalid type of entities: ' + str(type(entities)))
    assert all([isinstance(k, FakeDatastoreKey)
                for k in self._entities.keys()])

  @property
  def entities(self):
    """List of stored entities."""
    return self._entities

  def key(self, *args, **kwargs):
    """Creates datastore key."""
    return FakeDatastoreKey(*args, **kwargs)

  def entity(self, key):
    """Creates datastore entity."""
    assert isinstance(key, FakeDatastoreKey)
    return FakeDatastoreEntity(key)

  def no_transact_batch(self):
    """Starts batch of mutation which is committed without transaction."""
    return FakeDatastoreClientBatch(self)

  def transaction(self):
    """Starts datastore transaction."""
    result = FakeDatastoreClientTransaction(self)
    if self._transaction_hook:
      self._transaction_hook(self)
      self._transaction_hook = None
    return result

  def get(self, key, transaction=None):
    """Gets an entity with given key."""
    assert isinstance(key, FakeDatastoreKey)
    if transaction:
      return transaction.get(key)
    return copy.deepcopy(self._entities.get(key))

  def put(self, entity):
    """Updates entity in the datastore."""
    assert isinstance(entity, FakeDatastoreEntity)
    entity = copy.deepcopy(entity)
    if entity.key in self.entities:
      self.entities[entity.key].update(entity)
    else:
      self.entities[entity.key] = entity

  def batch(self):
    """Starts batch of mutations."""
    raise NotImplementedError('FakeDatastoreClient.batch not implemented')

  def query_fetch(self, **kwargs):
    """Queries datastore."""
    kind = kwargs.get('kind', None)
    ancestor = kwargs.get('ancestor', None)
    filters = kwargs.get('filters', [])
    if ancestor and not isinstance(ancestor, FakeDatastoreKey):
      raise ValueError('Invalid ancestor type: ' + str(type(ancestor)))
    if (('projection' in kwargs) or ('order' in kwargs) or
        ('distinct_on' in kwargs)):
      raise ValueError('Unsupported clause in arguments: ' + str(kwargs))
    for f in filters:
      if not isinstance(f, tuple) or len(f) != 3:
        raise ValueError('Invalid filter: ' + str(filters))
      if f[1] not in _QUERY_FILTER_OPERATOR.keys():
        raise ValueError('Unsupported operator in filters: ' + str(filters))
    for e in self._entities.values():
      key_tuple = e.key.flat_path
      if (kind is not None) and (key_tuple[-2] != kind):
        continue
      if (ancestor is not None) and (key_tuple[:-2] != ancestor.flat_path):
        continue
      all_filters_true = True
      for f in filters:
        if f[0] not in e:
          all_filters_true = False
          break
        if not _QUERY_FILTER_OPERATOR[f[1]](e[f[0]], f[2]):
          all_filters_true = False
          break
      if not all_filters_true:
        continue
      yield e

  def set_transaction_hook(self, hook):
    """Sets transaction hook.

    This hook will be executed right after next transaction created.
    It helps to model a situation when data are modified outside of transaction.
    To be used in tests to test how your code handles edits concurrent with
    transaction.

    Args:
      hook: transaction hook, should be a function which takes exactly
            one argument - instance of this class.

    Raises:
      ValueError: if transaction hook was already set
    """
    if self._transaction_hook is not None:
      raise ValueError('Attempt to set transaction hook twice')
    self._transaction_hook = hook

  def __str__(self):
    """Returns string representation of all stored entities."""
    buf = StringIO()
    for entity in self.entities.values():
      buf.write(u'Entity {0}:\n'.format(entity.key.flat_path))
      buf.write(u'    {0}\n'.format(dict(entity)))
    return buf.getvalue()
