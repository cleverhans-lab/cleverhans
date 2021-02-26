"""Tests for eval_lib.work_data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import unittest

from six import assertCountEqual
from six import itervalues

from eval_lib import work_data
from eval_lib.tests import fake_cloud_client


TEST_WORK_TYPE_ENTITY_ID = 'AllWork'


class WorkPiecesBaseTest(unittest.TestCase):

  def setUp(self):
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()
    self.work1 = {'submission_id': 's1',
                  'output_adversarial_batch_id': 'o1',
                  'claimed_worker_id': 'worker9999',
                  'claimed_worker_start_time': -1,
                  'is_completed': True}
    self.work2 = {'submission_id': 's2',
                  'output_adversarial_batch_id': 'o2',
                  'claimed_worker_id': None,
                  'claimed_worker_start_time': None,
                  'is_completed': False}

  def reset_work_pieces(self):
    self.work_pieces = work_data.WorkPiecesBase(self.datastore_client,
                                                TEST_WORK_TYPE_ENTITY_ID)

  def test_is_unclaimed(self):
    # completed work considered claimed
    self.assertFalse(work_data.is_unclaimed(self.work1))
    # not completed, not claimed work
    self.assertTrue(work_data.is_unclaimed(self.work2))
    # claimed but not completed work
    self.work2['claimed_worker_id'] = 'some_worker'
    self.work2['claimed_worker_start_time'] = work_data.get_integer_time()
    self.assertFalse(work_data.is_unclaimed(self.work2))
    # work claimed too long ago considered unclaimed now
    self.work2['claimed_worker_start_time'] = (
        work_data.get_integer_time() - work_data.MAX_PROCESSING_TIME - 1)
    self.assertTrue(work_data.is_unclaimed(self.work2))

  def test_write_to_datastore(self):
    self.reset_work_pieces()
    self.work_pieces.work['w1'] = self.work1
    self.work_pieces.work['w2'] = self.work2
    self.work_pieces.write_all_to_datastore()
    # verify content of the datastore
    parent_key = fake_cloud_client.FakeDatastoreKey(work_data.KIND_WORK_TYPE,
                                                    TEST_WORK_TYPE_ENTITY_ID)
    assertCountEqual(
        self, [fake_cloud_client.make_entity(parent_key)],
        self.datastore_client.query_fetch(kind=work_data.KIND_WORK_TYPE))
    entity1 = fake_cloud_client.make_entity(
        fake_cloud_client.FakeDatastoreKey(
            work_data.KIND_WORK, 'w1', parent=parent_key))
    entity1.update(self.work1)
    entity2 = fake_cloud_client.make_entity(
        fake_cloud_client.FakeDatastoreKey(
            work_data.KIND_WORK, 'w2', parent=parent_key))
    entity2.update(self.work2)
    assertCountEqual(
        self, [entity1, entity2],
        self.datastore_client.query_fetch(kind=work_data.KIND_WORK))

  def test_read_from_datastore(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    self.reset_work_pieces()
    self.work_pieces.read_all_from_datastore()
    # verify data
    self.assertDictEqual({'w10': self.work1, 'w20': self.work2},
                         self.work_pieces.work)

  def test_is_all_work_completed(self):
    self.reset_work_pieces()
    # empty set of work is considered completed
    self.assertTrue(self.work_pieces.is_all_work_competed())
    # one completed piece of work - all work completed
    self.work_pieces.work['w11'] = copy.deepcopy(self.work1)
    self.assertTrue(self.work_pieces.is_all_work_competed())
    # two completed pieces of work - all work completed
    self.work_pieces.work['w12'] = copy.deepcopy(self.work1)
    self.assertTrue(self.work_pieces.is_all_work_competed())
    # two completed and one incomplete pieces of work - work not completed
    self.work_pieces.work['w2'] = copy.deepcopy(self.work2)
    self.assertFalse(self.work_pieces.is_all_work_competed())

  def test_read_undone_from_datastore(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    self.reset_work_pieces()
    # return value is None because sharding is not used
    self.assertIsNone(self.work_pieces.read_undone_from_datastore())
    # Only work with ID 'w20' is undone
    self.assertDictEqual({'w20': self.work2}, self.work_pieces.work)

  def test_read_undone_from_datastore_same_shards(self):
    self.reset_work_pieces()
    self.work1['shard_id'] = 1
    self.work_pieces.work['w10'] = self.work1
    self.work2['shard_id'] = 2
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    self.reset_work_pieces()
    # return value is ID of the shard with undone work
    self.assertEqual(2, self.work_pieces.read_undone_from_datastore(
        shard_id=2, num_shards=3))
    # Only work with ID 'w20' is undone
    self.assertDictEqual({'w20': self.work2}, self.work_pieces.work)

  def test_read_undone_from_datastore_different_shards(self):
    self.reset_work_pieces()
    self.work1['shard_id'] = 1
    self.work_pieces.work['w10'] = self.work1
    self.work2['shard_id'] = 2
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    self.reset_work_pieces()
    # return value is ID of the shard with undone work
    self.assertEqual(2, self.work_pieces.read_undone_from_datastore(
        shard_id=1, num_shards=3))
    # Only work with ID 'w20' is undone
    self.assertDictEqual({'w20': self.work2}, self.work_pieces.work)

  def test_try_pick_piece_of_work_simple(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    work_id = self.work_pieces.try_pick_piece_of_work('worker0')
    self.assertEqual('w20', work_id)
    self.reset_work_pieces()
    self.work_pieces.read_all_from_datastore()
    self.assertEqual('worker0',
                     self.work_pieces.work['w20']['claimed_worker_id'])

  def test_try_pick_piece_of_work_all_completed(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.work['w20']['is_completed'] = True
    self.work_pieces.write_all_to_datastore()
    work_id = self.work_pieces.try_pick_piece_of_work('worker0')
    self.assertIsNone(work_id)

  def test_try_pick_piece_of_work_already_claimed(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work2['claimed_worker_id'] = 'other_worker'
    self.work2['claimed_worker_start_time'] = work_data.get_integer_time()
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    work_id = self.work_pieces.try_pick_piece_of_work('worker0')
    # if work is claimed by another worker then it won't be picked
    self.assertIsNone(work_id)

  def test_try_pick_piece_of_work_claimed_long_ago(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work2['claimed_worker_id'] = 'other_worker'
    self.work2['claimed_worker_start_time'] = (
        work_data.get_integer_time() - work_data.MAX_PROCESSING_TIME * 2)
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    work_id = self.work_pieces.try_pick_piece_of_work('worker0')
    # if work is claimed by another worker, but it happened some time ago
    # then work will be claimed
    self.assertEqual('w20', work_id)

  def test_try_pick_piece_of_work_concurrent_update(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    # any concurrent change in the entity will cause transaction to fail

    def transaction_hook(client):
      key = client.key('WorkType', TEST_WORK_TYPE_ENTITY_ID, 'Work', 'w20')
      client.entities[key]['output_adversarial_batch_id'] = 'o3'
    self.datastore_client.set_transaction_hook(transaction_hook)
    work_id = self.work_pieces.try_pick_piece_of_work('worker0')
    self.assertIsNone(work_id)

  def test_try_pick_piece_of_work_concurrent_update_of_other(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work_pieces.write_all_to_datastore()
    # concurrent change in entity which is not touched by the transaction
    # won't prevent transaction from completing

    def transaction_hook(client):
      key = client.key('WorkType', TEST_WORK_TYPE_ENTITY_ID, 'Work', 'w10')
      client.entities[key]['output_adversarial_batch_id'] = 'o3'
    self.datastore_client.set_transaction_hook(transaction_hook)
    work_id = self.work_pieces.try_pick_piece_of_work('worker0')
    self.assertEqual('w20', work_id)

  def test_update_work_as_completed(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work2['claimed_worker_id'] = 'this_worker'
    self.work2['claimed_worker_start_time'] = work_data.get_integer_time()
    self.work_pieces.write_all_to_datastore()
    self.assertTrue(
        self.work_pieces.update_work_as_completed('this_worker', 'w20'))
    self.reset_work_pieces()
    self.work_pieces.read_all_from_datastore()
    self.assertTrue(self.work_pieces.work['w20']['is_completed'])
    self.assertNotIn('error', self.work_pieces.work['w20'])

  def test_update_work_as_completed_other_values(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work2['claimed_worker_id'] = 'this_worker'
    self.work2['claimed_worker_start_time'] = work_data.get_integer_time()
    self.work_pieces.write_all_to_datastore()
    self.assertTrue(
        self.work_pieces.update_work_as_completed(
            'this_worker', 'w20', other_values={'a': 123, 'b': 456}))
    self.reset_work_pieces()
    self.work_pieces.read_all_from_datastore()
    self.assertTrue(self.work_pieces.work['w20']['is_completed'])
    self.assertNotIn('error', self.work_pieces.work['w20'])
    self.assertEqual(123, self.work_pieces.work['w20']['a'])
    self.assertEqual(456, self.work_pieces.work['w20']['b'])

  def test_update_work_as_completed_with_error(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work2['claimed_worker_id'] = 'this_worker'
    self.work2['claimed_worker_start_time'] = work_data.get_integer_time()
    self.work_pieces.write_all_to_datastore()
    self.assertTrue(
        self.work_pieces.update_work_as_completed(
            'this_worker', 'w20', error='err'))
    self.reset_work_pieces()
    self.work_pieces.read_all_from_datastore()
    self.assertTrue(self.work_pieces.work['w20']['is_completed'])
    self.assertEqual('err', self.work_pieces.work['w20']['error'])

  def test_update_work_as_completed_wrong_claimed_worker(self):
    self.reset_work_pieces()
    self.work_pieces.work['w10'] = self.work1
    self.work_pieces.work['w20'] = self.work2
    self.work2['claimed_worker_id'] = 'other_worker'
    self.work2['claimed_worker_start_time'] = work_data.get_integer_time()
    self.work_pieces.write_all_to_datastore()
    self.assertFalse(
        self.work_pieces.update_work_as_completed('this_worker', 'w20'))
    self.reset_work_pieces()
    self.work_pieces.read_all_from_datastore()
    self.assertFalse(self.work_pieces.work['w20']['is_completed'])

  def test_compute_work_stats(self):
    self.reset_work_pieces()
    self.work_pieces.work['w11'] = {
        'submission_id': 's1',
        'output_adversarial_batch_id': 'o1',
        'claimed_worker_id': 'worker1',
        'claimed_worker_start_time': -1,
        'is_completed': True,
        'elapsed_time': 1,
    }
    self.work_pieces.work['w12'] = {
        'submission_id': 's1',
        'output_adversarial_batch_id': 'o2',
        'claimed_worker_id': 'worker2',
        'claimed_worker_start_time': -1,
        'is_completed': False,
    }
    self.work_pieces.work['w21'] = {
        'submission_id': 's2',
        'output_adversarial_batch_id': 'o1',
        'claimed_worker_id': 'worker1',
        'claimed_worker_start_time': -1,
        'is_completed': True,
        'elapsed_time': 5,
    }
    self.work_pieces.work['w22'] = {
        'submission_id': 's2',
        'output_adversarial_batch_id': 'o2',
        'claimed_worker_id': 'worker2',
        'claimed_worker_start_time': -1,
        'is_completed': True,
        'elapsed_time': 10,
        'error': 'err',
    }
    self.work_pieces.work['w23'] = {
        'submission_id': 's2',
        'output_adversarial_batch_id': 'o1',
        'claimed_worker_id': 'worker1',
        'claimed_worker_start_time': -1,
        'is_completed': True,
        'elapsed_time': 7,
    }
    stats = self.work_pieces.compute_work_statistics()
    for v in itervalues(stats):
      v['eval_times'] = sorted(v['eval_times'])
    self.assertDictEqual(
        {
            's1': {'completed': 1,
                   'num_errors': 0,
                   'error_messages': set(),
                   'eval_times': [1.0],
                   'min_eval_time': 1.0,
                   'max_eval_time': 1.0,
                   'mean_eval_time': 1.0,
                   'median_eval_time': 1.0},
            's2': {'completed': 3,
                   'num_errors': 1,
                   'error_messages': set(['err']),
                   'eval_times': [5.0, 7.0],
                   'min_eval_time': 5.0,
                   'max_eval_time': 7.0,
                   'mean_eval_time': 6.0,
                   'median_eval_time': 6.0},
        }, stats)


class AttackWorkPiecesTest(unittest.TestCase):

  def setUp(self):
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()

  def test_init_from_adversarial_batches(self):
    adv_batches = {
        'ADVBATCH000': {'submission_id': 's1'},
        'ADVBATCH001': {'submission_id': 's2'},
        'ADVBATCH002': {'submission_id': 's3'},
    }
    expected_values = [
        {'claimed_worker_id': None, 'claimed_worker_start_time': None,
         'is_completed': False, 'error': None, 'elapsed_time': None,
         'submission_id': 's1', 'shard_id': None,
         'output_adversarial_batch_id': 'ADVBATCH000'},
        {'claimed_worker_id': None, 'claimed_worker_start_time': None,
         'is_completed': False, 'error': None, 'elapsed_time': None,
         'submission_id': 's2', 'shard_id': None,
         'output_adversarial_batch_id': 'ADVBATCH001'},
        {'claimed_worker_id': None, 'claimed_worker_start_time': None,
         'is_completed': False, 'error': None, 'elapsed_time': None,
         'submission_id': 's3', 'shard_id': None,
         'output_adversarial_batch_id': 'ADVBATCH002'}
    ]
    attack_work = work_data.AttackWorkPieces(self.datastore_client)
    attack_work.init_from_adversarial_batches(adv_batches)
    assertCountEqual(self, expected_values, attack_work.work.values())
    attack_work.write_all_to_datastore()
    attack_work = work_data.AttackWorkPieces(self.datastore_client)
    attack_work.read_all_from_datastore()
    assertCountEqual(self, expected_values, attack_work.work.values())


class DefenseWorkPiecesTest(unittest.TestCase):

  def setUp(self):
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()

  def test_init_from_classification_batches(self):
    class_batches = {
        'CBATCH000000': {'submission_id': 's1'},
        'CBATCH000001': {'submission_id': 's2'},
        'CBATCH000002': {'submission_id': 's3'},
    }
    expected_values = [
        {'claimed_worker_id': None, 'claimed_worker_start_time': None,
         'is_completed': False, 'error': None, 'elapsed_time': None,
         'submission_id': 's1', 'shard_id': None,
         'output_classification_batch_id': 'CBATCH000000'},
        {'claimed_worker_id': None, 'claimed_worker_start_time': None,
         'is_completed': False, 'error': None, 'elapsed_time': None,
         'submission_id': 's2', 'shard_id': None,
         'output_classification_batch_id': 'CBATCH000001'},
        {'claimed_worker_id': None, 'claimed_worker_start_time': None,
         'is_completed': False, 'error': None, 'elapsed_time': None,
         'submission_id': 's3', 'shard_id': None,
         'output_classification_batch_id': 'CBATCH000002'}
    ]
    defense_work = work_data.DefenseWorkPieces(self.datastore_client)
    defense_work.init_from_class_batches(class_batches)
    assertCountEqual(self, expected_values, defense_work.work.values())
    defense_work.write_all_to_datastore()
    defense_work = work_data.DefenseWorkPieces(self.datastore_client)
    defense_work.read_all_from_datastore()
    assertCountEqual(self, expected_values, defense_work.work.values())


if __name__ == '__main__':
  unittest.main()
