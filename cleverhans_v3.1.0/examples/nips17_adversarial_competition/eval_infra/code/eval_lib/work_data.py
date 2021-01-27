"""Module with classes to read and store data about work entities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import int # long in python 2

from io import StringIO
import pickle
import random
import time

import numpy as np

from six import iteritems
from six import itervalues
from six import text_type


# Cloud Datastore constants
KIND_WORK_TYPE = u'WorkType'
KIND_WORK = u'Work'
ID_ATTACKS_WORK_ENTITY = u'AllAttacks'
ID_DEFENSES_WORK_ENTITY = u'AllDefenses'

ATTACK_WORK_ID_PATTERN = u'WORKA{:03}'
DEFENSE_WORK_ID_PATTERN = u'WORKD{:05}'

# Constants for __str__
TO_STR_MAX_WORK = 20

# How long worker is allowed to process one piece of work,
# before considered failed
MAX_PROCESSING_TIME = 600

# Number of work records to read at once
MAX_WORK_RECORDS_READ = 1000


def get_integer_time():
  """Returns current time in long integer format."""
  return int(time.time())


def is_unclaimed(work):
  """Returns True if work piece is unclaimed."""
  if work['is_completed']:
    return False
  cutoff_time = time.time() - MAX_PROCESSING_TIME
  if (work['claimed_worker_id'] and
      work['claimed_worker_start_time'] is not None
      and work['claimed_worker_start_time'] >= cutoff_time):
    return False
  return True


class WorkPiecesBase(object):
  """Base class to store one piece of work.

  In adversarial competition, all work consists of the following:
  - evaluation of all attacks on all images from dataset which results in
    generation of adversarial images;
  - evaluation of all defenses on all adversarial images which results in
    storing classification labels.

  One piece of work is either evaluation of one attack on a subset of images or
  evaluation of one defense on a subset of adversarial images.
  This way all work is split into work pieces which could be computed
  independently in parallel by different workers.

  Each work piece is identified by unique ID and has one of the following
  statuses:
  - Unclaimed. This means that no worker has started working on the work piece.
  - Claimed by worker NN. This means that worker NN is working on this work
    piece. After workpiece being claimed for too long (more than
    MAX_PROCESSING_TIME seconds) it automatically considered unclaimed. This
    is needed in case worker failed while processing the work piece.
  - Completed. This means that computation of work piece is done.

  Additionally each work piece may be assigned to a shard. In such case
  workers are also grouped into shards. Each time worker looking for a work
  piece it first tries to find undone work from the shard worker is assigned to.
  Only after all work from this shard is done, worker will try to claim
  work pieces from other shards.

  The purpose of sharding is to reduce load on Google Cloud Datastore.
  """

  def __init__(self,
               datastore_client,
               work_type_entity_id):
    """Initializes WorkPiecesBase class.

    Args:
      datastore_client: instance of CompetitionDatastoreClient.
      work_type_entity_id: ID of the WorkType parent entity
    """
    self._datastore_client = datastore_client
    self._work_type_entity_id = work_type_entity_id
    # Dictionary: work_id -> dict with properties of the piece of work
    #
    # Common properties are following:
    # - claimed_worker_id - worker id which claimed the work
    # - claimed_worker_start_time - when work was claimed
    # - is_completed - whether work is completed or not
    # - error - if not None then work was completed with error
    # - elapsed_time - time took to complete the work
    # - shard_id - ID of the shard which run the work
    # - submission_id - ID of the submission which should be executed
    #
    # Additionally piece of work will have property specific to work type:
    # output_adversarial_batch_id for attack and output_classification_batch_id
    # for defense. Also upon completion of the work, worker may write
    # additional statistics field to the work.
    self._work = {}

  def serialize(self, fobj):
    """Serialize work pieces into file object."""
    pickle.dump(self._work, fobj)

  def deserialize(self, fobj):
    """Deserialize work pieces from file object."""
    self._work = pickle.load(fobj)

  @property
  def work(self):
    """Dictionary with all work pieces."""
    return self._work

  def replace_work(self, value):
    """Replaces work with provided value.

    Generally this method should be called only by master, that's why it
    separated from the property self.work.

    Args:
      value: dictionary with new work pieces
    """
    assert isinstance(value, dict)
    self._work = value

  def __len__(self):
    return len(self._work)

  def is_all_work_competed(self):
    """Returns whether all work pieces are completed or not."""
    return all([w['is_completed'] for w in itervalues(self.work)])

  def write_all_to_datastore(self):
    """Writes all work pieces into datastore.

    Each work piece is identified by ID. This method writes/updates only those
    work pieces which IDs are stored in this class. For examples, if this class
    has only work pieces with IDs  '1' ... '100' and datastore already contains
    work pieces with IDs '50' ... '200' then this method will create new
    work pieces with IDs '1' ... '49', update work pieces with IDs
    '50' ... '100' and keep unchanged work pieces with IDs '101' ... '200'.
    """
    client = self._datastore_client
    with client.no_transact_batch() as batch:
      parent_key = client.key(KIND_WORK_TYPE, self._work_type_entity_id)
      batch.put(client.entity(parent_key))
      for work_id, work_val in iteritems(self._work):
        entity = client.entity(client.key(KIND_WORK, work_id,
                                          parent=parent_key))
        entity.update(work_val)
        batch.put(entity)

  def read_all_from_datastore(self):
    """Reads all work pieces from the datastore."""
    self._work = {}
    client = self._datastore_client
    parent_key = client.key(KIND_WORK_TYPE, self._work_type_entity_id)
    for entity in client.query_fetch(kind=KIND_WORK, ancestor=parent_key):
      work_id = entity.key.flat_path[-1]
      self.work[work_id] = dict(entity)

  def _read_undone_shard_from_datastore(self, shard_id=None):
    """Reads undone worke pieces which are assigned to shard with given id."""
    self._work = {}
    client = self._datastore_client
    parent_key = client.key(KIND_WORK_TYPE, self._work_type_entity_id)
    filters = [('is_completed', '=', False)]
    if shard_id is not None:
      filters.append(('shard_id', '=', shard_id))
    for entity in client.query_fetch(kind=KIND_WORK, ancestor=parent_key,
                                     filters=filters):
      work_id = entity.key.flat_path[-1]
      self.work[work_id] = dict(entity)
      if len(self._work) >= MAX_WORK_RECORDS_READ:
        break

  def read_undone_from_datastore(self, shard_id=None, num_shards=None):
    """Reads undone work from the datastore.

    If shard_id and num_shards are specified then this method will attempt
    to read undone work for shard with id shard_id. If no undone work was found
    then it will try to read shard (shard_id+1) and so on until either found
    shard with undone work or all shards are read.

    Args:
      shard_id: Id of the start shard
      num_shards: total number of shards

    Returns:
      id of the shard with undone work which was read. None means that work
      from all datastore was read.
    """
    if shard_id is not None:
      shards_list = [(i + shard_id) % num_shards for i in range(num_shards)]
    else:
      shards_list = []
    shards_list.append(None)
    for shard in shards_list:
      self._read_undone_shard_from_datastore(shard)
      if self._work:
        return shard
    return None

  def try_pick_piece_of_work(self, worker_id, submission_id=None):
    """Tries pick next unclaimed piece of work to do.

    Attempt to claim work piece is done using Cloud Datastore transaction, so
    only one worker can claim any work piece at a time.

    Args:
      worker_id: ID of current worker
      submission_id: if not None then this method will try to pick
        piece of work for this submission

    Returns:
      ID of the claimed work piece
    """
    client = self._datastore_client
    unclaimed_work_ids = None
    if submission_id:
      unclaimed_work_ids = [
          k for k, v in iteritems(self.work)
          if is_unclaimed(v) and (v['submission_id'] == submission_id)
      ]
    if not unclaimed_work_ids:
      unclaimed_work_ids = [k for k, v in iteritems(self.work)
                            if is_unclaimed(v)]
    if unclaimed_work_ids:
      next_work_id = random.choice(unclaimed_work_ids)
    else:
      return None
    try:
      with client.transaction() as transaction:
        work_key = client.key(KIND_WORK_TYPE, self._work_type_entity_id,
                              KIND_WORK, next_work_id)
        work_entity = client.get(work_key, transaction=transaction)
        if not is_unclaimed(work_entity):
          return None
        work_entity['claimed_worker_id'] = worker_id
        work_entity['claimed_worker_start_time'] = get_integer_time()
        transaction.put(work_entity)
    except Exception:
      return None
    return next_work_id

  def update_work_as_completed(self, worker_id, work_id, other_values=None,
                               error=None):
    """Updates work piece in datastore as completed.

    Args:
      worker_id: ID of the worker which did the work
      work_id: ID of the work which was done
      other_values: dictionary with additonal values which should be saved
        with the work piece
      error: if not None then error occurred during computation of the work
        piece. In such case work will be marked as completed with error.

    Returns:
      whether work was successfully updated
    """
    client = self._datastore_client
    try:
      with client.transaction() as transaction:
        work_key = client.key(KIND_WORK_TYPE, self._work_type_entity_id,
                              KIND_WORK, work_id)
        work_entity = client.get(work_key, transaction=transaction)
        if work_entity['claimed_worker_id'] != worker_id:
          return False
        work_entity['is_completed'] = True
        if other_values:
          work_entity.update(other_values)
        if error:
          work_entity['error'] = text_type(error)
        transaction.put(work_entity)
    except Exception:
      return False
    return True

  def compute_work_statistics(self):
    """Computes statistics from all work pieces stored in this class."""
    result = {}
    for v in itervalues(self.work):
      submission_id = v['submission_id']
      if submission_id not in result:
        result[submission_id] = {
            'completed': 0,
            'num_errors': 0,
            'error_messages': set(),
            'eval_times': [],
            'min_eval_time': None,
            'max_eval_time': None,
            'mean_eval_time': None,
            'median_eval_time': None,
        }
      if not v['is_completed']:
        continue
      result[submission_id]['completed'] += 1
      if 'error' in v and v['error']:
        result[submission_id]['num_errors'] += 1
        result[submission_id]['error_messages'].add(v['error'])
      else:
        result[submission_id]['eval_times'].append(float(v['elapsed_time']))
    for v in itervalues(result):
      if v['eval_times']:
        v['min_eval_time'] = np.min(v['eval_times'])
        v['max_eval_time'] = np.max(v['eval_times'])
        v['mean_eval_time'] = np.mean(v['eval_times'])
        v['median_eval_time'] = np.median(v['eval_times'])
    return result

  def __str__(self):
    buf = StringIO()
    buf.write(u'WorkType "{0}"\n'.format(self._work_type_entity_id))
    for idx, (work_id, work_val) in enumerate(iteritems(self.work)):
      if idx >= TO_STR_MAX_WORK:
        buf.write(u'  ...\n')
        break
      buf.write(u'  Work "{0}"\n'.format(work_id))
      buf.write(u'    {0}\n'.format(str(work_val)))
    return buf.getvalue()


class AttackWorkPieces(WorkPiecesBase):
  """Subclass which represents work pieces for adversarial attacks."""

  def __init__(self, datastore_client):
    """Initializes AttackWorkPieces."""
    super(AttackWorkPieces, self).__init__(
        datastore_client=datastore_client,
        work_type_entity_id=ID_ATTACKS_WORK_ENTITY)

  def init_from_adversarial_batches(self, adv_batches):
    """Initializes work pieces from adversarial batches.

    Args:
      adv_batches: dict with adversarial batches,
        could be obtained as AversarialBatches.data
    """
    for idx, (adv_batch_id, adv_batch_val) in enumerate(iteritems(adv_batches)):
      work_id = ATTACK_WORK_ID_PATTERN.format(idx)
      self.work[work_id] = {
          'claimed_worker_id': None,
          'claimed_worker_start_time': None,
          'is_completed': False,
          'error': None,
          'elapsed_time': None,
          'submission_id': adv_batch_val['submission_id'],
          'shard_id': None,
          'output_adversarial_batch_id': adv_batch_id,
      }


class DefenseWorkPieces(WorkPiecesBase):
  """Subclass which represents work pieces for adversarial defenses."""

  def __init__(self, datastore_client):
    """Initializes DefenseWorkPieces."""
    super(DefenseWorkPieces, self).__init__(
        datastore_client=datastore_client,
        work_type_entity_id=ID_DEFENSES_WORK_ENTITY)

  def init_from_class_batches(self, class_batches, num_shards=None):
    """Initializes work pieces from classification batches.

    Args:
      class_batches: dict with classification batches, could be obtained
        as ClassificationBatches.data
      num_shards: number of shards to split data into,
        if None then no sharding is done.
    """
    shards_for_submissions = {}
    shard_idx = 0
    for idx, (batch_id, batch_val) in enumerate(iteritems(class_batches)):
      work_id = DEFENSE_WORK_ID_PATTERN.format(idx)
      submission_id = batch_val['submission_id']
      shard_id = None
      if num_shards:
        shard_id = shards_for_submissions.get(submission_id)
        if shard_id is None:
          shard_id = shard_idx % num_shards
          shards_for_submissions[submission_id] = shard_id
          shard_idx += 1
      # Note: defense also might have following fields populated by worker:
      # stat_correct, stat_error, stat_target_class, stat_num_images
      self.work[work_id] = {
          'claimed_worker_id': None,
          'claimed_worker_start_time': None,
          'is_completed': False,
          'error': None,
          'elapsed_time': None,
          'submission_id': submission_id,
          'shard_id': shard_id,
          'output_classification_batch_id': batch_id,
      }
