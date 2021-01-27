"""Tests for eval_lib.submissions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from six import assertCountEqual

from eval_lib import submissions
from eval_lib.tests import fake_cloud_client


ROUND_NAME = 'round-name'


class ParticipantFromSubmissionPathTest(unittest.TestCase):

  def test_team_id(self):
    self.assertDictEqual(
        {'team_id': 42},
        submissions.participant_from_submission_path('path/42.zip')
    )

  def test_baseline_id(self):
    self.assertDictEqual(
        {'baseline_id': 'a_1'},
        submissions.participant_from_submission_path('path/baseline_a_1.zip')
    )

  def test_tar_extension(self):
    self.assertDictEqual(
        {'team_id': 42},
        submissions.participant_from_submission_path('path/42.tar')
    )

  def test_tar_gz_extension(self):
    self.assertDictEqual(
        {'team_id': 42},
        submissions.participant_from_submission_path('path/42.tar.gz')
    )


class SubmissionsTest(unittest.TestCase):

  def setUp(self):
    storage_blobs = [
        ROUND_NAME + '/submissions/nontargeted/1.zip',
        ROUND_NAME + '/submissions/nontargeted/baseline_nt.zip',
        ROUND_NAME + '/submissions/targeted/1.zip',
        ROUND_NAME + '/submissions/targeted/2.zip',
        ROUND_NAME + '/submissions/defense/3.zip',
        ROUND_NAME + '/submissions/defense/baseline_adv_train.zip',
    ]
    self.storage_client = fake_cloud_client.FakeStorageClient(storage_blobs)
    self.datastore_client = fake_cloud_client.FakeDatastoreClient()
    self.submissions = submissions.CompetitionSubmissions(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        round_name=ROUND_NAME)

  def verify_submissions(self):
    assertCountEqual(self, [
        submissions.SubmissionDescriptor(
            path=(ROUND_NAME + '/submissions/nontargeted/1.zip'),
            participant_id={'team_id': 1}),
        submissions.SubmissionDescriptor(
            path=(ROUND_NAME + '/submissions/nontargeted/baseline_nt.zip'),
            participant_id={'baseline_id': 'nt'})
    ], self.submissions.attacks.values())
    assertCountEqual(self, [
        submissions.SubmissionDescriptor(
            path=(ROUND_NAME + '/submissions/targeted/1.zip'),
            participant_id={'team_id': 1}),
        submissions.SubmissionDescriptor(
            path=(ROUND_NAME + '/submissions/targeted/2.zip'),
            participant_id={'team_id': 2})
    ], self.submissions.targeted_attacks.values())
    assertCountEqual(self, [
        submissions.SubmissionDescriptor(
            path=(ROUND_NAME + '/submissions/defense/3.zip'),
            participant_id={'team_id': 3}),
        submissions.SubmissionDescriptor(
            path=(ROUND_NAME + '/submissions/defense/baseline_adv_train.zip'),
            participant_id={'baseline_id': 'adv_train'})
    ], self.submissions.defenses.values())
    self.assertEqual(len(self.submissions.attacks)
                     + len(self.submissions.targeted_attacks)
                     + len(self.submissions.defenses),
                     len(set(self.submissions.attacks.keys())
                         | set(self.submissions.targeted_attacks.keys())
                         | set(self.submissions.defenses.keys())))

  def verify_datastore_entities(self):
    # Verify 'SubmissionType' entities
    assertCountEqual(self, [
        self.datastore_client.entity(
            fake_cloud_client.FakeDatastoreKey('SubmissionType', 'Attacks')),
        self.datastore_client.entity(
            fake_cloud_client.FakeDatastoreKey('SubmissionType',
                                               'TargetedAttacks')),
        self.datastore_client.entity(
            fake_cloud_client.FakeDatastoreKey('SubmissionType', 'Defenses')),
    ], self.datastore_client.query_fetch(kind='SubmissionType'))
    # Verify 'Submission' entities
    expected_submission_entities = []
    for key_prefix, submission_entries in [
        (('SubmissionType', 'Attacks'), self.submissions.attacks),
        (('SubmissionType', 'TargetedAttacks'),
         self.submissions.targeted_attacks),
        (('SubmissionType', 'Defenses'), self.submissions.defenses)]:
      for k, v in submission_entries.items():
        entity = self.datastore_client.entity(
            fake_cloud_client.FakeDatastoreKey(
                *(key_prefix + ('Submission', k))))
        entity['submission_path'] = v.path
        entity.update(v.participant_id)
        expected_submission_entities.append(entity)
    assertCountEqual(self, expected_submission_entities,
                     self.datastore_client.query_fetch(kind='Submission'))

  def test_init_from_storage(self):
    self.submissions.init_from_storage_write_to_datastore()
    self.verify_submissions()
    self.verify_datastore_entities()

  def test_init_from_datastore(self):
    # first we need to populate datastore
    self.submissions.init_from_storage_write_to_datastore()
    # now reset submission class and load data from datastore
    self.submissions = submissions.CompetitionSubmissions(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        round_name=ROUND_NAME)
    self.assertFalse(self.submissions.attacks)
    self.assertFalse(self.submissions.targeted_attacks)
    self.assertFalse(self.submissions.defenses)
    self.submissions.init_from_datastore()
    self.verify_submissions()

  def test_get_all_attacks_ids(self):
    self.submissions.init_from_storage_write_to_datastore()
    # total will be two targeted and two not-targeted attacks,
    # their IDs are generated sequentially
    assertCountEqual(self, ['SUBA000', 'SUBA001', 'SUBT000', 'SUBT001'],
                     self.submissions.get_all_attack_ids())

  def test_find_by_id(self):
    self.submissions.init_from_storage_write_to_datastore()
    self.assertEqual(
        self.submissions.attacks['SUBA000'],
        self.submissions.find_by_id('SUBA000'))
    self.assertEqual(
        self.submissions.targeted_attacks['SUBT001'],
        self.submissions.find_by_id('SUBT001'))
    self.assertEqual(
        self.submissions.defenses['SUBD001'],
        self.submissions.find_by_id('SUBD001'))

  def test_get_external_id(self):
    self.submissions.init_from_storage_write_to_datastore()
    assertCountEqual(self, [3, 'baseline_adv_train'],
                     [self.submissions.get_external_id('SUBD000'),
                      self.submissions.get_external_id('SUBD001')])
    assertCountEqual(self, [1, 'baseline_nt'],
                     [self.submissions.get_external_id('SUBA000'),
                      self.submissions.get_external_id('SUBA001')])


if __name__ == '__main__':
  unittest.main()
