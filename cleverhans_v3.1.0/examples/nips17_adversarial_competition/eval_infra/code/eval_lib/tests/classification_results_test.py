"""Tests for eval_lib.classification_results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from six import assertCountEqual

from eval_lib import classification_results
from eval_lib import image_batches
from eval_lib import submissions
from eval_lib import work_data
from eval_lib.tests import fake_cloud_client


ROUND_NAME = 'round-name'


class FakeDatasetMeta(object):
  """Fake for DatasetMetadata which alwasy returns constants."""

  def get_true_label(self, _):
    return 1

  def get_target_class(self, _):
    return 2


class ClassificationResultsTest(unittest.TestCase):

  def setUp(self):
    self.storage_client = fake_cloud_client.FakeStorageClient()
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()
    self.submissions = submissions.CompetitionSubmissions(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        round_name=ROUND_NAME)
    # we only need list of submissin ids in CompetitionSubmissions for this test
    self.submissions._defenses = {
        'SUBD000': {},
        'SUBD001': {},
    }
    self.adv_batches = image_batches.AversarialBatches(
        datastore_client=self.datastore_client)
    self.adv_batches._data = {
        'ADVBATCH000': {'dataset_batch_id': 'BATCH000',
                        'images': {},
                        'submission_id': 'SUBA000'},
        'ADVBATCH001': {'dataset_batch_id': 'BATCH000',
                        'images': {},
                        'submission_id': 'SUBA001'},
        'ADVBATCH002': {'dataset_batch_id': 'BATCH000',
                        'images': {},
                        'submission_id': 'SUBT000'},
    }

  def verify_classification_batches(self, class_batches):
    assertCountEqual(
        self,
        [
            {'adversarial_batch_id': 'ADVBATCH000', 'submission_id': 'SUBD000',
             'result_path':
             (ROUND_NAME + '/classification_batches/SUBD000_ADVBATCH000.csv')},
            {'adversarial_batch_id': 'ADVBATCH000', 'submission_id': 'SUBD001',
             'result_path':
             (ROUND_NAME + '/classification_batches/SUBD001_ADVBATCH000.csv')},
            {'adversarial_batch_id': 'ADVBATCH001', 'submission_id': 'SUBD000',
             'result_path':
             (ROUND_NAME + '/classification_batches/SUBD000_ADVBATCH001.csv')},
            {'adversarial_batch_id': 'ADVBATCH001', 'submission_id': 'SUBD001',
             'result_path':
             (ROUND_NAME + '/classification_batches/SUBD001_ADVBATCH001.csv')},
            {'adversarial_batch_id': 'ADVBATCH002', 'submission_id': 'SUBD000',
             'result_path':
             (ROUND_NAME + '/classification_batches/SUBD000_ADVBATCH002.csv')},
            {'adversarial_batch_id': 'ADVBATCH002', 'submission_id': 'SUBD001',
             'result_path':
             (ROUND_NAME + '/classification_batches/SUBD001_ADVBATCH002.csv')},
        ], class_batches.data.values())

  def test_init_from_adv_batches_and_submissions(self):
    class_batches = classification_results.ClassificationBatches(
        self.datastore_client, self.storage_client, ROUND_NAME)
    class_batches.init_from_adversarial_batches_write_to_datastore(
        self.submissions, self.adv_batches)
    self.verify_classification_batches(class_batches)
    class_batches = classification_results.ClassificationBatches(
        self.datastore_client, self.storage_client, ROUND_NAME)
    class_batches.init_from_datastore()
    self.verify_classification_batches(class_batches)

  def test_read_batch_from_datastore(self):
    class_batches = classification_results.ClassificationBatches(
        self.datastore_client, self.storage_client, ROUND_NAME)
    class_batches.init_from_adversarial_batches_write_to_datastore(
        self.submissions, self.adv_batches)
    class_batches = classification_results.ClassificationBatches(
        self.datastore_client, self.storage_client, ROUND_NAME)
    # read first batch from datastore and verify that only one batch was read
    batch = class_batches.read_batch_from_datastore('CBATCH000000')
    self.assertEqual(0, len(class_batches.data))
    assertCountEqual(self, ['result_path', 'adversarial_batch_id',
                            'submission_id'], batch.keys())

  def test_compute_classification_results_from_defense_work(self):
    # Test computation of the results for the following case:
    # - one dataset batch BATCH000 with 5 images
    # - two defenses: SUBD000, SUBD001
    # - three attacks with corresponding adversarial batches:
    #      SUBA000 - ADVBATCH000
    #      SUBA001 - ADVBATCH001
    #      SUBT000 - ADVBATCH002
    #
    # Results are following (correct/incorrect/hit tc/total adv img):
    #            |      SUBD000     |      SUBD001     |
    #  ----------+------------------+------------------+
    #   SUBA000  | defense error    |  3 / 1 / 0 / 4   |
    #            |   WORK000        |   WORK001        |
    #  ----------+------------------+------------------+
    #   SUBA001  | 2 / 2 / 1 / 5    |  4 / 1 / 0 / 5   |
    #            |   WORK002        |   WORK003        |
    #  ----------+------------------+------------------+
    #   SUBT000  | 1 / 4 / 4 / 5    |  3 / 2 / 1 / 5   |
    #            |   WORK004        |   WORK005        |

    class_batches = classification_results.ClassificationBatches(
        self.datastore_client, self.storage_client, ROUND_NAME)
    result_path_prefix = ROUND_NAME + '/classification_batches/'
    class_batches._data = {
        'CBATCH000000': {
            'adversarial_batch_id': 'ADVBATCH000', 'submission_id': 'SUBD000',
            'result_path': result_path_prefix + 'SUBD000_ADVBATCH000.csv'},
        'CBATCH000001': {
            'adversarial_batch_id': 'ADVBATCH000', 'submission_id': 'SUBD001',
            'result_path': result_path_prefix + 'SUBD001_ADVBATCH000.csv'},
        'CBATCH000002': {
            'adversarial_batch_id': 'ADVBATCH001', 'submission_id': 'SUBD000',
            'result_path': result_path_prefix + 'SUBD000_ADVBATCH001.csv'},
        'CBATCH000003': {
            'adversarial_batch_id': 'ADVBATCH001', 'submission_id': 'SUBD001',
            'result_path': result_path_prefix + 'SUBD001_ADVBATCH001.csv'},
        'CBATCH000004': {
            'adversarial_batch_id': 'ADVBATCH002', 'submission_id': 'SUBD000',
            'result_path': result_path_prefix + 'SUBD000_ADVBATCH002.csv'},
        'CBATCH000005': {
            'adversarial_batch_id': 'ADVBATCH002', 'submission_id': 'SUBD001',
            'result_path': result_path_prefix + 'SUBD001_ADVBATCH002.csv'},
    }
    defense_work = work_data.DefenseWorkPieces(self.datastore_client)
    defense_work._work = {
        'WORK000': {'output_classification_batch_id': 'CBATCH000000',
                    'error': 'error'},
        'WORK001': {'output_classification_batch_id': 'CBATCH000001',
                    'stat_correct': 3, 'stat_error': 1, 'stat_target_class': 0,
                    'stat_num_images': 4, 'error': None},
        'WORK002': {'output_classification_batch_id': 'CBATCH000002',
                    'stat_correct': 2, 'stat_error': 2, 'stat_target_class': 1,
                    'stat_num_images': 5, 'error': None},
        'WORK003': {'output_classification_batch_id': 'CBATCH000003',
                    'stat_correct': 4, 'stat_error': 1, 'stat_target_class': 0,
                    'stat_num_images': 5, 'error': None},
        'WORK004': {'output_classification_batch_id': 'CBATCH000004',
                    'stat_correct': 1, 'stat_error': 4, 'stat_target_class': 4,
                    'stat_num_images': 5, 'error': None},
        'WORK005': {'output_classification_batch_id': 'CBATCH000005',
                    'stat_correct': 3, 'stat_error': 2, 'stat_target_class': 1,
                    'stat_num_images': 5, 'error': None},
    }
    # Compute and verify results
    (accuracy_matrix, error_matrix, hit_target_class_matrix,
     processed_images_count) = class_batches.compute_classification_results(
         self.adv_batches, dataset_batches=None, dataset_meta=None,
         defense_work=defense_work)
    self.assertDictEqual(
        {
            ('SUBD001', 'SUBA000'): 3,
            ('SUBD000', 'SUBA001'): 2,
            ('SUBD001', 'SUBA001'): 4,
            ('SUBD000', 'SUBT000'): 1,
            ('SUBD001', 'SUBT000'): 3,
        },
        accuracy_matrix._items)
    self.assertDictEqual(
        {
            ('SUBD001', 'SUBA000'): 1,
            ('SUBD000', 'SUBA001'): 2,
            ('SUBD001', 'SUBA001'): 1,
            ('SUBD000', 'SUBT000'): 4,
            ('SUBD001', 'SUBT000'): 2,
        },
        error_matrix._items)
    self.assertDictEqual(
        {
            ('SUBD001', 'SUBA000'): 0,
            ('SUBD000', 'SUBA001'): 1,
            ('SUBD001', 'SUBA001'): 0,
            ('SUBD000', 'SUBT000'): 4,
            ('SUBD001', 'SUBT000'): 1,
        },
        hit_target_class_matrix._items)
    self.assertDictEqual({'SUBD000': 10, 'SUBD001': 14},
                         processed_images_count)

  def test_read_classification_results(self):
    self.storage_client = fake_cloud_client.FakeStorageClient(
        {'filename': 'img1.png,123\nimg2.jpg,456'})
    results = classification_results.read_classification_results(
        self.storage_client, 'filename')
    self.assertDictEqual({'img1': 123, 'img2': 456}, results)

  def test_analyze_one_classification_result(self):
    self.storage_client = fake_cloud_client.FakeStorageClient(
        {'filename':
         'a1.png,1\na2.png,4\na3.png,1\na4.png,1\na5.png,2\na6.png,9'})
    adv_batch = {
        'dataset_batch_id': 'BATCH000',
        'images': {'a' + str(i):
                   {'clean_image_id': 'c' + str(i)} for i in range(1, 6)}
    }
    dataset_batches = image_batches.DatasetBatches(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        dataset_name='final')
    dataset_batches._data = {
        'BATCH000': {'images': {'c' + str(i): {'dataset_image_id': str(i)}
                                for i in range(1, 6)}},
    }
    (count_correctly_classified, count_errors,
     count_hit_target_class, num_images) = (
         classification_results.analyze_one_classification_result(
             self.storage_client, 'filename', adv_batch, dataset_batches,
             FakeDatasetMeta()))
    self.assertEqual(3, count_correctly_classified)
    self.assertEqual(2, count_errors)
    self.assertEqual(1, count_hit_target_class)
    self.assertEqual(5, num_images)


if __name__ == '__main__':
  unittest.main()
