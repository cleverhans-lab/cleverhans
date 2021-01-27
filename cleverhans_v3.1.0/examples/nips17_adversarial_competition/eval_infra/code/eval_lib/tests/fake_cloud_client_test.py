"""Tests for eval_lib.testing.fake_cloud_client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from io import BytesIO
import unittest
from eval_lib.tests import fake_cloud_client
from six import assertCountEqual
from six import b as six_b


class FakeStorageClientTest(unittest.TestCase):

  def test_list_blobs(self):
    all_blobs = [
        'some_blob',
        'dataset/dev_dataset.csv',
        'dataset/dev/img1.png',
        'dataset/dev/img2.png'
    ]
    client = fake_cloud_client.FakeStorageClient(all_blobs)
    assertCountEqual(self, all_blobs, client.list_blobs())
    assertCountEqual(self, [
        'dataset/dev_dataset.csv',
        'dataset/dev/img1.png',
        'dataset/dev/img2.png'
    ], client.list_blobs('dataset/dev'))
    assertCountEqual(self, [
        'dataset/dev/img1.png',
        'dataset/dev/img2.png'
    ], client.list_blobs('dataset/dev/'))

  def test_get_blob(self):
    client = fake_cloud_client.FakeStorageClient({'some_blob': 'some_content',
                                                  'blob2': 'another_content'})
    self.assertIsNone(client.get_blob('blob3'))
    buf = BytesIO()
    client.get_blob('some_blob').download_to_file(buf)
    self.assertEqual(six_b('some_content'), buf.getvalue())


class FakeDatastoreKeyTest(unittest.TestCase):

  def test_flat_path(self):
    key1 = fake_cloud_client.FakeDatastoreKey('abc', '1')
    self.assertTupleEqual(('abc', '1'), key1.flat_path)
    key2 = fake_cloud_client.FakeDatastoreKey('def', 'xyz', parent=key1)
    self.assertTupleEqual(('abc', '1', 'def', 'xyz'), key2.flat_path)

  def test_equality(self):
    key1a = fake_cloud_client.FakeDatastoreKey('abc', '1')
    key1b = fake_cloud_client.FakeDatastoreKey('abc', '1')
    key2a = fake_cloud_client.FakeDatastoreKey('def', 'xyz', parent=key1a)
    key2b = fake_cloud_client.FakeDatastoreKey('def', 'xyz', parent=key1a)
    # key equal to self
    self.assertTrue(key1a == key1a)
    self.assertFalse(key1a != key1a)
    # key equal to the same key
    self.assertTrue(key1a == key1b)
    self.assertFalse(key1a != key1b)
    self.assertTrue(key2a == key2b)
    self.assertFalse(key2a != key2b)
    # key different from other key
    self.assertFalse(key1a == key2a)
    self.assertTrue(key1a != key2a)
    # key not equal to tuple
    self.assertTrue(key1a != key1a.flat_path)
    self.assertFalse(key1a == key1a.flat_path)


class FakeDatastoreEntityTest(unittest.TestCase):

  def test_key(self):
    entity = fake_cloud_client.make_entity(('abc', '1'))
    self.assertEqual(entity.key,
                     fake_cloud_client.FakeDatastoreKey('abc', '1'))

  def test_equality_keys(self):
    entity1a = fake_cloud_client.make_entity(('abc', '1'))
    entity1b = fake_cloud_client.make_entity(('abc', '1'))
    entity2 = fake_cloud_client.make_entity(('abc', '2'))
    self.assertFalse(entity1a == entity2)
    self.assertTrue(entity1a != entity2)
    self.assertTrue(entity1a == entity1b)
    self.assertFalse(entity1b != entity1b)

  def test_equality_dict(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1'))
    entity1['k1'] = 'v1'
    entity2 = fake_cloud_client.make_entity(('abc', '1'))
    entity2['k1'] = 'v2'
    entity3 = fake_cloud_client.make_entity(('abc', '1'))
    entity1['k1'] = 'v1'
    entity1['k2'] = 'v2'
    # compare to self
    self.assertTrue(entity1 == entity1)
    self.assertFalse(entity1 != entity1)
    self.assertTrue(entity2 == entity2)
    self.assertFalse(entity2 != entity2)
    self.assertTrue(entity3 == entity3)
    self.assertFalse(entity3 != entity3)
    # compare to others
    self.assertFalse(entity1 == entity2)
    self.assertTrue(entity1 != entity2)
    self.assertFalse(entity1 == entity3)
    self.assertTrue(entity1 != entity3)
    self.assertFalse(entity2 == entity3)
    self.assertTrue(entity2 != entity3)

  def test_copy(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1'))
    entity1['k1'] = ['v1']
    self.assertEqual(entity1.key,
                     fake_cloud_client.FakeDatastoreKey('abc', '1'))
    self.assertEqual(dict(entity1),
                     {'k1': ['v1']})
    entity2 = copy.copy(entity1)
    entity2['k1'].append('v2')
    entity2['k3'] = 'v3'
    self.assertIsInstance(entity2, fake_cloud_client.FakeDatastoreEntity)
    self.assertEqual(entity1.key,
                     fake_cloud_client.FakeDatastoreKey('abc', '1'))
    self.assertEqual(dict(entity1),
                     {'k1': ['v1', 'v2']})
    self.assertEqual(entity2.key,
                     fake_cloud_client.FakeDatastoreKey('abc', '1'))
    self.assertEqual(dict(entity2),
                     {'k1': ['v1', 'v2'], 'k3': 'v3'})

  def test_deep_copy(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1'))
    entity1['k1'] = ['v1']
    self.assertEqual(entity1.key,
                     fake_cloud_client.FakeDatastoreKey('abc', '1'))
    self.assertEqual(dict(entity1),
                     {'k1': ['v1']})
    entity2 = copy.deepcopy(entity1)
    entity2['k1'].append('v2')
    entity2['k3'] = 'v3'
    self.assertIsInstance(entity2, fake_cloud_client.FakeDatastoreEntity)
    self.assertEqual(entity1.key,
                     fake_cloud_client.FakeDatastoreKey('abc', '1'))
    self.assertEqual(dict(entity1),
                     {'k1': ['v1']})
    self.assertEqual(entity2.key,
                     fake_cloud_client.FakeDatastoreKey('abc', '1'))
    self.assertEqual(dict(entity2),
                     {'k1': ['v1', 'v2'], 'k3': 'v3'})


class FakeDatastoreClientTest(unittest.TestCase):

  def setUp(self):
    self._client = fake_cloud_client.FakeDatastoreClient()
    self._key1 = self._client.key('abc', 'def')
    self._key2 = self._client.key('qwe', 'rty', parent=self._key1)
    self._entity1 = self._client.entity(self._key1)
    self._entity1['k1'] = 'v1'
    self._entity2 = self._client.entity(self._key2)
    self._entity2['k2'] = 'v2'
    self._entity2['k3'] = 'v3'

  def test_make_key(self):
    self.assertTupleEqual(('abc', 'def'), self._key1.flat_path)
    self.assertTupleEqual(('abc', 'def', 'qwe', 'rty'), self._key2.flat_path)

  def test_make_entity(self):
    self.assertTupleEqual(('abc', 'def'), self._entity1.key.flat_path)

  def test_put_entity(self):
    self.assertDictEqual({}, self._client.entities)
    self._client.put(self._entity1)
    self.assertDictEqual({self._key1: self._entity1}, self._client.entities)
    self._client.put(self._entity2)
    self.assertDictEqual({self._key1: self._entity1, self._key2: self._entity2},
                         self._client.entities)

  def test_get_entity(self):
    self._client.put(self._entity1)
    self._client.put(self._entity2)
    self.assertEqual(self._entity1, self._client.get(self._key1))
    self.assertEqual(self._entity2, self._client.get(self._key2))

  def test_write_batch(self):
    with self._client.no_transact_batch() as batch:
      batch.put(self._entity1)
      batch.put(self._entity2)
    assertCountEqual(self, [self._key1, self._key2],
                     self._client.entities.keys())
    self.assertEqual(self._key1, self._client.entities[self._key1].key)
    self.assertDictEqual({'k1': 'v1'}, dict(self._client.entities[self._key1]))
    self.assertEqual(self._key2, self._client.entities[self._key2].key)
    self.assertDictEqual({'k2': 'v2', 'k3': 'v3'},
                         dict(self._client.entities[self._key2]))

  def test_overwrite_values(self):
    client = fake_cloud_client.FakeDatastoreClient()
    key1 = client.key('abc', 'def')
    entity1 = client.entity(key1)
    entity1['k1'] = 'v1'
    entity2 = client.entity(key1)
    entity2['k1'] = 'v2'
    entity2['k2'] = 'v3'
    with client.no_transact_batch() as batch:
      batch.put(entity1)
    assertCountEqual(self, [key1], client.entities.keys())
    self.assertEqual(key1, client.entities[key1].key)
    self.assertDictEqual({'k1': 'v1'}, dict(client.entities[key1]))
    with client.no_transact_batch() as batch:
      batch.put(entity2)
    assertCountEqual(self, [key1], client.entities.keys())
    self.assertEqual(key1, client.entities[key1].key)
    self.assertDictEqual({'k1': 'v2', 'k2': 'v3'}, dict(client.entities[key1]))

  def test_query_fetch_all(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1'))
    entity1['k1'] = 'v1'
    entity2 = fake_cloud_client.make_entity(('abc', '1', 'def', '2'))
    entity2['k2'] = 'v2'
    client = fake_cloud_client.FakeDatastoreClient([entity1, entity2])
    assertCountEqual(self, [entity1, entity2], client.query_fetch())

  def test_query_fetch_kind_filter(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1'))
    entity1['k1'] = 'v1'
    entity2 = fake_cloud_client.make_entity(('abc', '1', 'def', '2'))
    entity2['k2'] = 'v2'
    client = fake_cloud_client.FakeDatastoreClient([entity1, entity2])
    assertCountEqual(self, [entity1], client.query_fetch(kind='abc'))
    assertCountEqual(self, [entity2], client.query_fetch(kind='def'))

  def test_query_fetch_ancestor_filter(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1', 'def', '2'))
    entity1['k1'] = 'v1'
    entity2 = fake_cloud_client.make_entity(('xyz', '3', 'qwe', '4'))
    entity2['k2'] = 'v2'
    client = fake_cloud_client.FakeDatastoreClient([entity1, entity2])
    assertCountEqual(self, [entity1],
                     client.query_fetch(ancestor=client.key('abc', '1')))
    assertCountEqual(self, [entity2],
                     client.query_fetch(ancestor=client.key('xyz', '3')))

  def test_query_fetch_ancestor_and_kind_filter(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1', 'def', '2'))
    entity1['k1'] = 'v1'
    entity2 = fake_cloud_client.make_entity(('abc', '1', 'xyz', '3'))
    entity2['k2'] = 'v2'
    entity3 = fake_cloud_client.make_entity(('def', '4'))
    entity3['k2'] = 'v2'
    client = fake_cloud_client.FakeDatastoreClient([entity1, entity2, entity3])
    assertCountEqual(self, [entity1],
                     client.query_fetch(kind='def',
                                        ancestor=client.key('abc', '1')))

  def test_query_fetch_data_filter(self):
    entity1 = fake_cloud_client.make_entity(('abc', '1'))
    entity1['k1'] = 'v1'
    entity2 = fake_cloud_client.make_entity(('abc', '2'))
    entity2['k1'] = 'v2'
    entity2['k2'] = 'v2'
    entity3 = fake_cloud_client.make_entity(('abc', '3'))
    entity3['k2'] = 'v3'
    client = fake_cloud_client.FakeDatastoreClient([entity1, entity2, entity3])
    assertCountEqual(self, [entity1],
                     client.query_fetch(filters=[('k1', '=', 'v1')]))
    assertCountEqual(self, [entity2],
                     client.query_fetch(filters=[('k1', '>', 'v1')]))
    assertCountEqual(self, [entity1, entity2],
                     client.query_fetch(filters=[('k1', '>=', 'v1')]))
    assertCountEqual(self, [entity2],
                     client.query_fetch(filters=[('k2', '<', 'v3')]))
    assertCountEqual(self, [entity2, entity3],
                     client.query_fetch(filters=[('k2', '<=', 'v3')]))
    assertCountEqual(self, [entity2],
                     client.query_fetch(filters=[('k1', '>=', 'v1'),
                                                 ('k2', '<=', 'v3')]))


class FakeDatastoreClientTransactionTest(unittest.TestCase):

  def setUp(self):
    self._client = fake_cloud_client.FakeDatastoreClient()
    self._key1 = self._client.key('abc', 'def')
    self._key2 = self._client.key('qwe', 'rty', parent=self._key1)
    self._key3 = self._client.key('123', '456')
    self._entity1 = self._client.entity(self._key1)
    self._entity1['k1'] = 'v1'
    self._entity2 = self._client.entity(self._key2)
    self._entity2['k2'] = 'v2'
    self._entity2['k3'] = 'v3'
    self._entity3 = self._client.entity(self._key3)
    self._entity3['k4'] = 'v4'
    self._entity3['k5'] = 'v5'
    self._entity3['k6'] = 'v6'
    self._client.put(self._entity1)
    self._client.put(self._entity2)
    self._client.put(self._entity3)
    # verify datastore content
    assertCountEqual(self, [self._key1, self._key2, self._key3],
                     self._client.entities.keys())
    self.assertDictEqual({'k1': 'v1'}, dict(self._client.entities[self._key1]))
    self.assertDictEqual({'k2': 'v2', 'k3': 'v3'},
                         dict(self._client.entities[self._key2]))
    self.assertDictEqual({'k4': 'v4', 'k5': 'v5', 'k6': 'v6'},
                         dict(self._client.entities[self._key3]))

  def test_transaction_write_only_no_concurrent(self):
    key4 = self._client.key('zxc', 'vbn')
    entity4 = self._client.entity(key4)
    entity4['k7'] = 'v7'
    entity3_upd = self._client.entity(self._key3)
    entity3_upd['k4'] = 'upd_v4'
    with self._client.transaction() as transaction:
      # first write in transaction
      transaction.put(entity4)
      # second write in transaction
      transaction.put(entity3_upd)
    # verify datastore content
    assertCountEqual(self, [self._key1, self._key2, self._key3, key4],
                     self._client.entities.keys())
    self.assertDictEqual({'k1': 'v1'}, dict(self._client.entities[self._key1]))
    self.assertDictEqual({'k2': 'v2', 'k3': 'v3'},
                         dict(self._client.entities[self._key2]))
    self.assertDictEqual({'k4': 'upd_v4', 'k5': 'v5', 'k6': 'v6'},
                         dict(self._client.entities[self._key3]))
    self.assertDictEqual({'k7': 'v7'}, dict(self._client.entities[key4]))

  def test_transaction_read_write_no_concurrent(self):
    key4 = self._client.key('zxc', 'vbn')
    entity4 = self._client.entity(key4)
    entity4['k7'] = 'v7'
    entity3_upd = self._client.entity(self._key3)
    entity3_upd['k4'] = 'upd_v4'
    with self._client.transaction() as transaction:
      # reading in transaction always returns data snapshot before transaction
      read_entity = self._client.get(self._key3, transaction=transaction)
      self.assertDictEqual({'k4': 'v4', 'k5': 'v5', 'k6': 'v6'},
                           dict(read_entity))
      # first write in transaction
      transaction.put(entity3_upd)
      # second write in transaction
      transaction.put(entity4)
      # reading in transaction always returns data snapshot before transaction
      read_entity = self._client.get(self._key3, transaction=transaction)
      self.assertDictEqual({'k4': 'v4', 'k5': 'v5', 'k6': 'v6'},
                           dict(read_entity))
    # verify datastore content
    assertCountEqual(self, [self._key1, self._key2, self._key3, key4],
                     self._client.entities.keys())
    self.assertDictEqual({'k1': 'v1'}, dict(self._client.entities[self._key1]))
    self.assertDictEqual({'k2': 'v2', 'k3': 'v3'},
                         dict(self._client.entities[self._key2]))
    self.assertDictEqual({'k4': 'upd_v4', 'k5': 'v5', 'k6': 'v6'},
                         dict(self._client.entities[self._key3]))
    self.assertDictEqual({'k7': 'v7'}, dict(self._client.entities[key4]))

  def test_transaction_read_write_concurrent_not_intersecting(self):
    key4 = self._client.key('zxc', 'vbn')
    entity4 = self._client.entity(key4)
    entity4['k7'] = 'v7'
    entity3_upd = self._client.entity(self._key3)
    entity3_upd['k4'] = 'upd_v4'
    entity1_upd = self._client.entity(self._key1)
    entity1_upd['k1'] = 'upd_v1'
    with self._client.transaction() as transaction:
      # reading in transaction always returns data snapshot before transaction
      read_entity = self._client.get(self._key3, transaction=transaction)
      self.assertDictEqual({'k4': 'v4', 'k5': 'v5', 'k6': 'v6'},
                           dict(read_entity))
      # first write in transaction
      transaction.put(entity3_upd)
      # modify some data which are not references in the transaction
      self._client.put(entity1_upd)
      # second write in transaction
      transaction.put(entity4)
      # reading in transaction always returns data snapshot before transaction
      read_entity = self._client.get(self._key3, transaction=transaction)
      self.assertDictEqual({'k4': 'v4', 'k5': 'v5', 'k6': 'v6'},
                           dict(read_entity))
    # verify datastore content
    assertCountEqual(self, [self._key1, self._key2, self._key3, key4],
                     self._client.entities.keys())
    self.assertDictEqual({'k1': 'upd_v1'},
                         dict(self._client.entities[self._key1]))
    self.assertDictEqual({'k2': 'v2', 'k3': 'v3'},
                         dict(self._client.entities[self._key2]))
    self.assertDictEqual({'k4': 'upd_v4', 'k5': 'v5', 'k6': 'v6'},
                         dict(self._client.entities[self._key3]))
    self.assertDictEqual({'k7': 'v7'}, dict(self._client.entities[key4]))

  def test_transaction_write_concurrent(self):
    key4 = self._client.key('zxc', 'vbn')
    entity4 = self._client.entity(key4)
    entity4['k7'] = 'v7'
    entity3_upd = self._client.entity(self._key3)
    entity3_upd['k4'] = 'upd_v4'
    entity3_upd_no_transact = self._client.entity(self._key3)
    entity3_upd_no_transact['k4'] = 'another_v4'
    reached_end_of_transaction = False
    with self.assertRaises(Exception):
      with self._client.transaction() as transaction:
        # first write in transaction
        transaction.put(entity3_upd)
        # modify some data which are not references in the transaction
        self._client.put(entity3_upd_no_transact)
        # second write in transaction
        transaction.put(entity4)
        reached_end_of_transaction = True
    self.assertTrue(reached_end_of_transaction)
    # verify datastore content
    assertCountEqual(self, [self._key1, self._key2, self._key3],
                     self._client.entities.keys())
    self.assertDictEqual({'k1': 'v1'}, dict(self._client.entities[self._key1]))
    self.assertDictEqual({'k2': 'v2', 'k3': 'v3'},
                         dict(self._client.entities[self._key2]))
    self.assertDictEqual({'k4': 'another_v4', 'k5': 'v5', 'k6': 'v6'},
                         dict(self._client.entities[self._key3]))

  def test_transaction_read_concurrent(self):
    key4 = self._client.key('zxc', 'vbn')
    entity4 = self._client.entity(key4)
    entity4['k7'] = 'v7'
    entity3_upd_no_transact = self._client.entity(self._key3)
    entity3_upd_no_transact['k4'] = 'another_v4'
    reached_end_of_transaction = False
    with self.assertRaises(Exception):
      with self._client.transaction() as transaction:
        # write in transaction
        transaction.put(entity4)
        # read in transaction
        read_entity = self._client.get(self._key3, transaction=transaction)
        self.assertDictEqual({'k4': 'v4', 'k5': 'v5', 'k6': 'v6'},
                             dict(read_entity))
        # modify some data which are not references in the transaction
        self._client.put(entity3_upd_no_transact)
        reached_end_of_transaction = True
    self.assertTrue(reached_end_of_transaction)
    # verify datastore content
    assertCountEqual(self, [self._key1, self._key2, self._key3],
                     self._client.entities.keys())
    self.assertDictEqual({'k1': 'v1'}, dict(self._client.entities[self._key1]))
    self.assertDictEqual({'k2': 'v2', 'k3': 'v3'},
                         dict(self._client.entities[self._key2]))
    self.assertDictEqual({'k4': 'another_v4', 'k5': 'v5', 'k6': 'v6'},
                         dict(self._client.entities[self._key3]))


if __name__ == '__main__':
  unittest.main()
