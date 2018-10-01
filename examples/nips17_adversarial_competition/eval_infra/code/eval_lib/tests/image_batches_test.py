"""Tests for eval_lib.image_batches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import unittest

from six import assertCountEqual

from eval_lib import image_batches
from eval_lib import submissions
from eval_lib.tests import fake_cloud_client


ROUND_NAME = 'round-name'


class ImageBatchesBaseTest(unittest.TestCase):

  def setUp(self):
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()
    self.image_batches = image_batches.ImageBatchesBase(
        datastore_client=self.datastore_client,
        entity_kind_batches='Batch',
        entity_kind_images='Image')

  def test_add_batch(self):
    self.assertEqual(0, len(self.image_batches.data))
    self.image_batches.add_batch('batch1',
                                 batch_properties={'k1': 'v1', 'k2': 'v2'})
    self.assertEqual(1, len(self.image_batches.data))
    self.assertDictEqual({'k1': 'v1', 'k2': 'v2', 'images': {}},
                         self.image_batches['batch1'])
    self.image_batches.add_batch('batch2', batch_properties={'k3': 'v3'})
    self.assertEqual(2, len(self.image_batches.data))
    self.assertDictEqual({'k3': 'v3', 'images': {}},
                         self.image_batches['batch2'])

  def test_add_image(self):
    self.assertEqual(0, len(self.image_batches.data))
    self.image_batches.add_batch('batch1',
                                 batch_properties={'k1': 'v1', 'k2': 'v2'})
    self.image_batches.add_image('batch1', 'img1',
                                 image_properties={'k4': 'v4'})
    self.assertEqual(1, len(self.image_batches.data))
    self.assertDictEqual({'k1': 'v1', 'k2': 'v2',
                          'images': {'img1': {'k4': 'v4'}}},
                         self.image_batches['batch1'])
    self.image_batches.add_image('batch1', 'img2',
                                 image_properties={'k5': 'v5'})
    self.assertEqual(1, len(self.image_batches.data))
    self.assertDictEqual({'k1': 'v1', 'k2': 'v2',
                          'images': {'img1': {'k4': 'v4'},
                                     'img2': {'k5': 'v5'}}},
                         self.image_batches['batch1'])

  def test_write_to_datastore(self):
    # add 2 batches and 3 images, write everything to datastore
    self.image_batches.add_batch('batch1',
                                 batch_properties={'k1': 'v1', 'k2': 'v2'})
    self.image_batches.add_batch('batch2', batch_properties={'k3': 'v3'})
    self.image_batches.add_image('batch1', 'img1',
                                 image_properties={'k4': 'v4'})
    self.image_batches.add_image('batch1', 'img2',
                                 image_properties={'k5': 'v5'})
    self.image_batches.add_image('batch2', 'img3',
                                 image_properties={'k6': 'v6'})
    self.image_batches.write_to_datastore()
    # verify batches
    batch_entity1 = self.datastore_client.entity(
        fake_cloud_client.FakeDatastoreKey('Batch', 'batch1'))
    batch_entity1.update({'k1': 'v1', 'k2': 'v2'})
    batch_entity2 = self.datastore_client.entity(
        fake_cloud_client.FakeDatastoreKey('Batch', 'batch2'))
    batch_entity2.update({'k3': 'v3'})
    assertCountEqual(self, [batch_entity1, batch_entity2],
                     self.datastore_client.query_fetch(kind='Batch'))
    # verify images
    img_entity1 = self.datastore_client.entity(
        fake_cloud_client.FakeDatastoreKey('Batch', 'batch2', 'Image', 'img1'))
    img_entity1.update({'k4': 'v4'})
    img_entity2 = self.datastore_client.entity(
        fake_cloud_client.FakeDatastoreKey('Batch', 'batch2', 'Image', 'img2'))
    img_entity2.update({'k5': 'v5'})
    img_entity3 = self.datastore_client.entity(
        fake_cloud_client.FakeDatastoreKey('Batch', 'batch2', 'Image', 'img3'))
    img_entity3.update({'k6': 'v6'})

  def test_write_single_batch_images_to_datastore(self):
    # add 2 batches and 3 images, write only one batch to datastore
    self.image_batches.add_batch('batch1',
                                 batch_properties={'k1': 'v1', 'k2': 'v2'})
    self.image_batches.add_batch('batch2', batch_properties={'k3': 'v3'})
    self.image_batches.add_image('batch1', 'img1',
                                 image_properties={'k4': 'v4'})
    self.image_batches.add_image('batch1', 'img2',
                                 image_properties={'k5': 'v5'})
    self.image_batches.add_image('batch2', 'img3',
                                 image_properties={'k6': 'v6'})
    self.image_batches.write_single_batch_images_to_datastore('batch2')
    # verify batches
    # write_single_batch_images_to_datastore writes only images, so no batches
    assertCountEqual(self, [], self.datastore_client.query_fetch(kind='Batch'))
    # verify images
    img_entity3 = self.datastore_client.entity(
        fake_cloud_client.FakeDatastoreKey('Batch', 'batch2', 'Image', 'img3'))
    img_entity3.update({'k6': 'v6'})
    assertCountEqual(self, [img_entity3],
                     self.datastore_client.query_fetch(kind='Image'))


class DatasetBatchesTest(unittest.TestCase):

  def setUp(self):
    storage_blobs = [
        'dataset/dev/img1.png',
        'dataset/dev/img2.png',
        'dataset/dev/img3.png',
        'dataset/dev/img4.png',
        'dataset/dev/img5.png',
        'dataset/dev_dataset.csv',
    ]
    self.storage_client = fake_cloud_client.FakeStorageClient(storage_blobs)
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()
    self.dataset_batches = image_batches.DatasetBatches(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        dataset_name='dev')

  def verify_dataset_batches(self):
    self.assertEqual(2, len(self.dataset_batches.data))
    all_images = {}
    for batch in self.dataset_batches.data.values():
      self.assertIn(batch['epsilon'], [4, 8, 12, 16])
      self.assertGreaterEqual(3, len(batch['images']))
      self.assertTrue(
          set(all_images.keys()).isdisjoint(batch['images'].keys()),
          msg=('all_images and batch[\'images\'] contains common keys %s'
               % set(all_images.keys()).intersection(batch['images'].keys()))
      )
      all_images.update(batch['images'])
    assertCountEqual(self, [
        {'dataset_image_id': 'img1', 'image_path': 'dataset/dev/img1.png'},
        {'dataset_image_id': 'img2', 'image_path': 'dataset/dev/img2.png'},
        {'dataset_image_id': 'img3', 'image_path': 'dataset/dev/img3.png'},
        {'dataset_image_id': 'img4', 'image_path': 'dataset/dev/img4.png'},
        {'dataset_image_id': 'img5', 'image_path': 'dataset/dev/img5.png'},
    ], all_images.values())

  def verify_datastore_entities(self):
    # Verify 'DatasetBatch' entities
    expected_batch_entities = []
    for batch_id, batch in self.dataset_batches.data.items():
      entity = self.datastore_client.entity(
          fake_cloud_client.FakeDatastoreKey('DatasetBatch', batch_id))
      entity['epsilon'] = batch['epsilon']
      expected_batch_entities.append(entity)
    assertCountEqual(self, expected_batch_entities,
                     self.datastore_client.query_fetch(kind='DatasetBatch'))
    # Verify 'DatasetImage' entities
    expected_image_entities = []
    for batch_id, batch in self.dataset_batches.data.items():
      for image_id, image in batch['images'].items():
        entity = self.datastore_client.entity(
            fake_cloud_client.FakeDatastoreKey('DatasetBatch', batch_id,
                                               'DatasetImage', image_id))
        entity.update(image)
        expected_image_entities.append(entity)
    assertCountEqual(self, expected_image_entities,
                     self.datastore_client.query_fetch(kind='DatasetImage'))

  def test_init_from_storage(self):
    self.dataset_batches.init_from_storage_write_to_datastore(batch_size=3)
    self.verify_dataset_batches()
    self.verify_datastore_entities()

  def test_init_from_datastore(self):
    self.dataset_batches.init_from_storage_write_to_datastore(batch_size=3)
    self.dataset_batches = image_batches.DatasetBatches(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        dataset_name='dev')
    self.dataset_batches.init_from_datastore()
    self.verify_dataset_batches()

  def test_count_num_images(self):
    self.dataset_batches.init_from_storage_write_to_datastore(batch_size=3)
    self.assertEqual(5, self.dataset_batches.count_num_images())


class AdversarialBatchesTest(unittest.TestCase):

  def setUp(self):
    # prepare dataset batches and submissions
    storage_blobs = [
        'dataset/dev/img1.png',
        'dataset/dev/img2.png',
        'dataset/dev/img3.png',
        'dataset/dev/img4.png',
        'dataset/dev/img5.png',
        'dataset/dev_dataset.csv',
        ROUND_NAME + '/submissions/nontargeted/1.zip',
        ROUND_NAME + '/submissions/nontargeted/baseline_nt.zip',
        ROUND_NAME + '/submissions/targeted/1.zip',
        ROUND_NAME + '/submissions/targeted/2.zip',
        ROUND_NAME + '/submissions/defense/3.zip',
        ROUND_NAME + '/submissions/defense/baseline_adv_train.zip',
    ]
    self.storage_client = fake_cloud_client.FakeStorageClient(storage_blobs)
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()
    self.dataset_batches = image_batches.DatasetBatches(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        dataset_name='dev')
    self.dataset_batches.init_from_storage_write_to_datastore(batch_size=3)
    self.submissions = submissions.CompetitionSubmissions(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        round_name=ROUND_NAME)
    self.submissions.init_from_storage_write_to_datastore()

  def verify_adversarial_batches_without_images(self, adv_batches):
    attack_ids = (list(self.submissions.attacks.keys())
                  + list(self.submissions.targeted_attacks.keys()))
    dataset_batch_ids = self.dataset_batches.data.keys()
    expected_batches = [
        {'dataset_batch_id': b_id, 'submission_id': a_id, 'images': {}}
        for (b_id, a_id) in itertools.product(dataset_batch_ids, attack_ids)
    ]
    assertCountEqual(self, expected_batches, adv_batches.data.values())

  def test_init_from_dataset_and_submissions(self):
    adv_batches = image_batches.AversarialBatches(
        datastore_client=self.datastore_client)
    adv_batches.init_from_dataset_and_submissions_write_to_datastore(
        dataset_batches=self.dataset_batches,
        attack_submission_ids=self.submissions.get_all_attack_ids())
    self.verify_adversarial_batches_without_images(adv_batches)

  def test_init_from_datastore(self):
    # populate datastore
    adv_batches = image_batches.AversarialBatches(
        datastore_client=self.datastore_client)
    adv_batches.init_from_dataset_and_submissions_write_to_datastore(
        dataset_batches=self.dataset_batches,
        attack_submission_ids=self.submissions.get_all_attack_ids())
    # init AversarialBatches from datastore
    adv_batches = image_batches.AversarialBatches(
        datastore_client=self.datastore_client)
    adv_batches.init_from_datastore()
    self.verify_adversarial_batches_without_images(adv_batches)

  def test_count_generated_adv_examples(self):
    adv_batches = image_batches.AversarialBatches(
        datastore_client=self.datastore_client)
    adv_batches.init_from_dataset_and_submissions_write_to_datastore(
        dataset_batches=self.dataset_batches,
        attack_submission_ids=self.submissions.get_all_attack_ids())
    self.assertDictEqual(
        {'SUBA000': 0, 'SUBA001': 0, 'SUBT000': 0, 'SUBT001': 0},
        adv_batches.count_generated_adv_examples())


if __name__ == '__main__':
  unittest.main()
