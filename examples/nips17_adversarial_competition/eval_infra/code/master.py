"""Master which prepares work for all workers.

Evaluation of competition is split into work pieces. One work piece is a
either evaluation of an attack on a batch of images or evaluation of a
defense on a batch of adversarial images.
Work pieces are run by workers. Master prepares work pieces for workers and
writes them to the datastore.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from collections import defaultdict
import csv
from io import BytesIO
import logging
import os
import pickle
import random
import time

from six import iteritems
from six import iterkeys
from six import itervalues
from six.moves import input as input_str

import eval_lib


# List of allowed sizes of adversarial perturbation
ALLOWED_EPS = [4, 8, 12, 16]

# Batch size
DEFAULT_BATCH_SIZE = 100


def print_header(text):
  """Prints header with given text and frame composed of '#' characters."""
  print()
  print('#'*(len(text)+4))
  print('# ' + text + ' #')
  print('#'*(len(text)+4))
  print()


def save_dict_to_file(filename, dictionary):
  """Saves dictionary as CSV file."""
  with open(filename, 'w') as f:
    writer = csv.writer(f)
    for k, v in iteritems(dictionary):
      writer.writerow([str(k), str(v)])


class EvaluationMaster(object):
  """Class which encapsulates logit of the master."""

  def __init__(self, storage_client, datastore_client, round_name, dataset_name,
               blacklisted_submissions='', results_dir='',
               num_defense_shards=None, verbose=False,
               batch_size=DEFAULT_BATCH_SIZE, max_dataset_num_images=None):
    """Initializes EvaluationMaster.

    Args:
      storage_client: instance of eval_lib.CompetitionStorageClient
      datastore_client: instance of eval_lib.CompetitionDatastoreClient
      round_name: name of the current round
      dataset_name: name of the dataset, 'dev' or 'final'
      blacklisted_submissions: optional list of blacklisted submissions which
        should not be evaluated
      results_dir: local directory where results and logs should be written
      num_defense_shards: optional number of defense shards
      verbose: whether output should be verbose on not. If True, then methods
        of this class will print some additional information which is useful
        for debugging.
      batch_size: batch size to use
      max_dataset_num_images: maximum number of images from the dataset to use
        or None if entire dataset should be used.
    """
    self.storage_client = storage_client
    self.datastore_client = datastore_client
    self.round_name = round_name
    self.dataset_name = dataset_name
    self.results_dir = results_dir
    if num_defense_shards:
      self.num_defense_shards = int(num_defense_shards)
    else:
      self.num_defense_shards = None
    self.verbose = verbose
    self.blacklisted_submissions = [s.strip()
                                    for s in blacklisted_submissions.split(',')]
    self.batch_size = batch_size
    self.max_dataset_num_images = max_dataset_num_images
    # init client classes
    self.submissions = eval_lib.CompetitionSubmissions(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        round_name=self.round_name)
    self.dataset_batches = eval_lib.DatasetBatches(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        dataset_name=self.dataset_name)
    self.adv_batches = eval_lib.AversarialBatches(
        datastore_client=self.datastore_client)
    self.class_batches = eval_lib.ClassificationBatches(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        round_name=self.round_name)
    self.attack_work = eval_lib.AttackWorkPieces(
        datastore_client=self.datastore_client)
    self.defense_work = eval_lib.DefenseWorkPieces(
        datastore_client=self.datastore_client)

  def ask_when_work_is_populated(self, work):
    """When work is already populated asks whether we should continue.

    This method prints warning message that work is populated and asks
    whether user wants to continue or not.

    Args:
      work: instance of WorkPiecesBase

    Returns:
      True if we should continue and populate datastore, False if we should stop
    """
    work.read_all_from_datastore()
    if work.work:
      print('Work is already written to datastore.\n'
            'If you continue these data will be overwritten and '
            'possible corrupted.')
      inp = input_str('Do you want to continue? '
                      '(type "yes" without quotes to confirm): ')
      return inp == 'yes'
    else:
      return True

  def prepare_attacks(self):
    """Prepares all data needed for evaluation of attacks."""
    print_header('PREPARING ATTACKS DATA')
    # verify that attacks data not written yet
    if not self.ask_when_work_is_populated(self.attack_work):
      return
    self.attack_work = eval_lib.AttackWorkPieces(
        datastore_client=self.datastore_client)
    # prepare submissions
    print_header('Initializing submissions')
    self.submissions.init_from_storage_write_to_datastore()
    if self.verbose:
      print(self.submissions)
    # prepare dataset batches
    print_header('Initializing dataset batches')
    self.dataset_batches.init_from_storage_write_to_datastore(
        batch_size=self.batch_size,
        allowed_epsilon=ALLOWED_EPS,
        skip_image_ids=[],
        max_num_images=self.max_dataset_num_images)
    if self.verbose:
      print(self.dataset_batches)
    # prepare adversarial batches
    print_header('Initializing adversarial batches')
    self.adv_batches.init_from_dataset_and_submissions_write_to_datastore(
        dataset_batches=self.dataset_batches,
        attack_submission_ids=self.submissions.get_all_attack_ids())
    if self.verbose:
      print(self.adv_batches)
    # prepare work pieces
    print_header('Preparing attack work pieces')
    self.attack_work.init_from_adversarial_batches(self.adv_batches.data)
    self.attack_work.write_all_to_datastore()
    if self.verbose:
      print(self.attack_work)

  def prepare_defenses(self):
    """Prepares all data needed for evaluation of defenses."""
    print_header('PREPARING DEFENSE DATA')
    # verify that defense data not written yet
    if not self.ask_when_work_is_populated(self.defense_work):
      return
    self.defense_work = eval_lib.DefenseWorkPieces(
        datastore_client=self.datastore_client)
    # load results of attacks
    self.submissions.init_from_datastore()
    self.dataset_batches.init_from_datastore()
    self.adv_batches.init_from_datastore()
    self.attack_work.read_all_from_datastore()
    # populate classification results
    print_header('Initializing classification batches')
    self.class_batches.init_from_adversarial_batches_write_to_datastore(
        self.submissions, self.adv_batches)
    if self.verbose:
      print(self.class_batches)
    # populate work pieces
    print_header('Preparing defense work pieces')
    self.defense_work.init_from_class_batches(
        self.class_batches.data, num_shards=self.num_defense_shards)
    self.defense_work.write_all_to_datastore()
    if self.verbose:
      print(self.defense_work)

  def _save_work_results(self, run_stats, scores, num_processed_images,
                         filename):
    """Saves statistics about each submission.

    Saved statistics include score; number of completed and failed batches;
    min, max, average and median time needed to run one batch.

    Args:
      run_stats: dictionary with runtime statistics for submissions,
        can be generated by WorkPiecesBase.compute_work_statistics
      scores: dictionary mapping submission ids to scores
      num_processed_images: dictionary with number of successfully processed
        images by each submission, one of the outputs of
        ClassificationBatches.compute_classification_results
      filename: output filename
    """
    with open(filename, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(
          ['SubmissionID', 'ExternalSubmissionId', 'Score',
           'CompletedBatches', 'BatchesWithError', 'ProcessedImages',
           'MinEvalTime', 'MaxEvalTime',
           'MedianEvalTime', 'MeanEvalTime',
           'ErrorMsg'])
      for submission_id in sorted(iterkeys(run_stats)):
        stat = run_stats.get(
            submission_id,
            collections.defaultdict(lambda: float('NaN')))
        external_id = self.submissions.get_external_id(submission_id)
        error_msg = ''
        while not error_msg and stat['error_messages']:
          error_msg = stat['error_messages'].pop()
          if error_msg.startswith('Cant copy adversarial batch locally'):
            error_msg = ''
        writer.writerow([
            submission_id, external_id, scores.get(submission_id, None),
            stat['completed'], stat['num_errors'],
            num_processed_images.get(submission_id, None),
            stat['min_eval_time'], stat['max_eval_time'],
            stat['median_eval_time'], stat['mean_eval_time'],
            error_msg
        ])

  def _save_sorted_results(self, run_stats, scores, image_count, filename):
    """Saves sorted (by score) results of the evaluation.

    Args:
      run_stats: dictionary with runtime statistics for submissions,
        can be generated by WorkPiecesBase.compute_work_statistics
      scores: dictionary mapping submission ids to scores
      image_count: dictionary with number of images processed by submission
      filename: output filename
    """
    with open(filename, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['SubmissionID', 'ExternalTeamId', 'Score',
                       'MedianTime', 'ImageCount'])

      def get_second(x):
        """Returns second entry of a list/tuple"""
        return x[1]
      for s_id, score in sorted(iteritems(scores),
                                key=get_second, reverse=True):
        external_id = self.submissions.get_external_id(s_id)
        stat = run_stats.get(
            s_id, collections.defaultdict(lambda: float('NaN')))
        writer.writerow([s_id, external_id, score,
                         stat['median_eval_time'],
                         image_count[s_id]])

  def _read_dataset_metadata(self):
    """Reads dataset metadata.

    Returns:
      instance of DatasetMetadata
    """
    blob = self.storage_client.get_blob(
        'dataset/' + self.dataset_name + '_dataset.csv')
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return eval_lib.DatasetMetadata(buf)

  def compute_results(self):
    """Computes results (scores, stats, etc...) of competition evaluation.

    Results are saved into output directory (self.results_dir).
    Also this method saves all intermediate data into output directory as well,
    so it can resume computation if it was interrupted for some reason.
    This is useful because computatin of resuls could take many minutes.
    """
    # read all data
    logging.info('Reading data from datastore')
    dataset_meta = self._read_dataset_metadata()
    self.submissions.init_from_datastore()
    self.dataset_batches.init_from_datastore()
    self.adv_batches.init_from_datastore()
    self.attack_work.read_all_from_datastore()

    if os.path.exists(os.path.join(self.results_dir, 'defense_work.dump')):
      with open(os.path.join(self.results_dir, 'defense_work.dump')) as f:
        self.defense_work.deserialize(f)
    else:
      self.defense_work.read_all_from_datastore()
      with open(os.path.join(self.results_dir, 'defense_work.dump'), 'w') as f:
        self.defense_work.serialize(f)

    if os.path.exists(os.path.join(self.results_dir, 'class_batches.dump')):
      with open(os.path.join(self.results_dir, 'class_batches.dump')) as f:
        self.class_batches.deserialize(f)
    else:
      self.class_batches.init_from_datastore()
      with open(os.path.join(self.results_dir, 'class_batches.dump'), 'w') as f:
        self.class_batches.serialize(f)

    # process data
    logging.info('Processing classification results')
    count_adv_images = self.adv_batches.count_generated_adv_examples()
    intermediate_files = ['acc_matrix.dump', 'error_matrix.dump',
                          'hit_tc_matrix.dump', 'classified_images_count.dump']
    if all([os.path.exists(os.path.join(self.results_dir, fname))
            for fname in intermediate_files]):
      with open(os.path.join(self.results_dir, 'acc_matrix.dump')) as f:
        acc_matrix = pickle.load(f)
      with open(os.path.join(self.results_dir, 'error_matrix.dump')) as f:
        error_matrix = pickle.load(f)
      with open(os.path.join(self.results_dir, 'hit_tc_matrix.dump')) as f:
        hit_tc_matrix = pickle.load(f)
      with open(os.path.join(self.results_dir,
                             'classified_images_count.dump')) as f:
        classified_images_count = pickle.load(f)
    else:
      acc_matrix, error_matrix, hit_tc_matrix, classified_images_count = (
          self.class_batches.compute_classification_results(
              self.adv_batches,
              self.dataset_batches,
              dataset_meta,
              self.defense_work))
      with open(os.path.join(self.results_dir, 'acc_matrix.dump'), 'w') as f:
        pickle.dump(acc_matrix, f)
      with open(os.path.join(self.results_dir, 'error_matrix.dump'), 'w') as f:
        pickle.dump(error_matrix, f)
      with open(os.path.join(self.results_dir, 'hit_tc_matrix.dump'), 'w') as f:
        pickle.dump(hit_tc_matrix, f)
      with open(os.path.join(self.results_dir,
                             'classified_images_count.dump'), 'w') as f:
        pickle.dump(classified_images_count, f)

    # compute attacks and defenses which will be used for scoring
    logging.info('Computing attacks and defenses which are used for scoring')
    expected_num_adv_images = self.dataset_batches.count_num_images()
    attacks_to_use = [k for k, v in iteritems(count_adv_images)
                      if ((v == expected_num_adv_images)
                          and (k not in self.blacklisted_submissions))]

    total_num_adversarial = sum(itervalues(count_adv_images))
    defenses_to_use = [k for k, v in iteritems(classified_images_count)
                       if ((v == total_num_adversarial)
                           and (k not in self.blacklisted_submissions))]

    logging.info('Expected number of adversarial images: %d',
                 expected_num_adv_images)
    logging.info('Number of attacks to use to score defenses: %d',
                 len(attacks_to_use))
    logging.info('Expected number of classification predictions: %d',
                 total_num_adversarial)
    logging.info('Number of defenses to use to score attacks: %d',
                 len(defenses_to_use))

    save_dict_to_file(os.path.join(self.results_dir, 'count_adv_images.csv'),
                      count_adv_images)
    save_dict_to_file(os.path.join(self.results_dir,
                                   'classified_images_count.csv'),
                      classified_images_count)

    # compute scores
    logging.info('Computing scores')
    attack_scores = defaultdict(lambda: 0)
    targeted_attack_scores = defaultdict(lambda: 0)
    defense_scores = defaultdict(lambda: 0)
    for defense_id in acc_matrix.dim0:
      for attack_id in acc_matrix.dim1:
        if attack_id in attacks_to_use:
          defense_scores[defense_id] += acc_matrix[defense_id, attack_id]
        if defense_id in defenses_to_use:
          if attack_id in self.submissions.targeted_attacks:
            targeted_attack_scores[attack_id] += (
                hit_tc_matrix[defense_id, attack_id])
          else:
            attack_scores[attack_id] += error_matrix[defense_id, attack_id]
    # negate results of blacklisted submissions
    for s_id in self.blacklisted_submissions:
      if s_id in defense_scores:
        defense_scores[s_id] = -defense_scores[s_id]
      if s_id in attack_scores:
        attack_scores[s_id] = -attack_scores[s_id]
      if s_id in targeted_attack_scores:
        targeted_attack_scores[s_id] = -targeted_attack_scores[s_id]
    # save results
    logging.info('Saving results')
    all_attack_stats = self.attack_work.compute_work_statistics()
    nontargeted_attack_stats = {k: v for k, v in iteritems(all_attack_stats)
                                if k in self.submissions.attacks}
    targeted_attack_stats = {k: v for k, v in iteritems(all_attack_stats)
                             if k in self.submissions.targeted_attacks}
    defense_stats = self.defense_work.compute_work_statistics()
    self._save_work_results(
        nontargeted_attack_stats, attack_scores, count_adv_images,
        os.path.join(self.results_dir, 'attack_results.csv'))
    self._save_work_results(
        targeted_attack_stats, targeted_attack_scores, count_adv_images,
        os.path.join(self.results_dir, 'targeted_attack_results.csv'))
    self._save_work_results(
        defense_stats, defense_scores,
        classified_images_count,
        os.path.join(self.results_dir, 'defense_results.csv'))

    self._save_sorted_results(
        nontargeted_attack_stats, attack_scores, count_adv_images,
        os.path.join(self.results_dir, 'sorted_attack_results.csv'))
    self._save_sorted_results(
        targeted_attack_stats, targeted_attack_scores, count_adv_images,
        os.path.join(self.results_dir, 'sorted_target_attack_results.csv'))
    self._save_sorted_results(
        defense_stats, defense_scores, classified_images_count,
        os.path.join(self.results_dir, 'sorted_defense_results.csv'))

    defense_id_to_name = {k: self.submissions.get_external_id(k)
                          for k in iterkeys(self.submissions.defenses)}
    attack_id_to_name = {k: self.submissions.get_external_id(k)
                         for k in self.submissions.get_all_attack_ids()}
    acc_matrix.save_to_file(
        os.path.join(self.results_dir, 'accuracy_matrix.csv'),
        remap_dim0=defense_id_to_name, remap_dim1=attack_id_to_name)
    error_matrix.save_to_file(
        os.path.join(self.results_dir, 'error_matrix.csv'),
        remap_dim0=defense_id_to_name, remap_dim1=attack_id_to_name)
    hit_tc_matrix.save_to_file(
        os.path.join(self.results_dir, 'hit_target_class_matrix.csv'),
        remap_dim0=defense_id_to_name, remap_dim1=attack_id_to_name)

    save_dict_to_file(os.path.join(self.results_dir, 'defense_id_to_name.csv'),
                      defense_id_to_name)
    save_dict_to_file(os.path.join(self.results_dir, 'attack_id_to_name.csv'),
                      attack_id_to_name)

  def _show_status_for_work(self, work):
    """Shows status for given work pieces.

    Args:
      work: instance of either AttackWorkPieces or DefenseWorkPieces
    """
    work_count = len(work.work)
    work_completed = {}
    work_completed_count = 0
    for v in itervalues(work.work):
      if v['is_completed']:
        work_completed_count += 1
        worker_id = v['claimed_worker_id']
        if worker_id not in work_completed:
          work_completed[worker_id] = {
              'completed_count': 0,
              'last_update': 0.0,
          }
        work_completed[worker_id]['completed_count'] += 1
        work_completed[worker_id]['last_update'] = max(
            work_completed[worker_id]['last_update'],
            v['claimed_worker_start_time'])
    print('Completed {0}/{1} work'.format(work_completed_count,
                                          work_count))
    for k in sorted(iterkeys(work_completed)):
      last_update_time = time.strftime(
          '%Y-%m-%d %H:%M:%S',
          time.localtime(work_completed[k]['last_update']))
      print('Worker {0}: completed {1}   last claimed work at {2}'.format(
          k, work_completed[k]['completed_count'], last_update_time))

  def _export_work_errors(self, work, output_file):
    """Saves errors for given work pieces into file.

    Args:
      work: instance of either AttackWorkPieces or DefenseWorkPieces
      output_file: name of the output file
    """
    errors = set()
    for v in itervalues(work.work):
      if v['is_completed'] and v['error'] is not None:
        errors.add(v['error'])
    with open(output_file, 'w') as f:
      for e in sorted(errors):
        f.write(e)
        f.write('\n')

  def show_status(self):
    """Shows current status of competition evaluation.

    Also this method saves error messages generated by attacks and defenses
    into attack_errors.txt and defense_errors.txt.
    """
    print_header('Attack work statistics')
    self.attack_work.read_all_from_datastore()
    self._show_status_for_work(self.attack_work)
    self._export_work_errors(
        self.attack_work,
        os.path.join(self.results_dir, 'attack_errors.txt'))
    print_header('Defense work statistics')
    self.defense_work.read_all_from_datastore()
    self._show_status_for_work(self.defense_work)
    self._export_work_errors(
        self.defense_work,
        os.path.join(self.results_dir, 'defense_errors.txt'))

  def cleanup_failed_attacks(self):
    """Cleans up data of failed attacks."""
    print_header('Cleaning up failed attacks')
    attacks_to_replace = {}
    self.attack_work.read_all_from_datastore()
    failed_submissions = set()
    error_msg = set()
    for k, v in iteritems(self.attack_work.work):
      if v['error'] is not None:
        attacks_to_replace[k] = dict(v)
        failed_submissions.add(v['submission_id'])
        error_msg.add(v['error'])
        attacks_to_replace[k].update(
            {
                'claimed_worker_id': None,
                'claimed_worker_start_time': None,
                'is_completed': False,
                'error': None,
                'elapsed_time': None,
            })
    self.attack_work.replace_work(attacks_to_replace)
    print('Affected submissions:')
    print(' '.join(sorted(failed_submissions)))
    print('Error messages:')
    print(' '.join(sorted(error_msg)))
    print('')
    inp = input_str('Are you sure? (type "yes" without quotes to confirm): ')
    if inp != 'yes':
      return
    self.attack_work.write_all_to_datastore()
    print('Work cleaned up')

  def cleanup_attacks_with_zero_images(self):
    """Cleans up data about attacks which generated zero images."""
    print_header('Cleaning up attacks which generated 0 images.')
    # find out attack work to cleanup
    self.adv_batches.init_from_datastore()
    self.attack_work.read_all_from_datastore()
    new_attack_work = {}
    affected_adversarial_batches = set()
    for work_id, work in iteritems(self.attack_work.work):
      adv_batch_id = work['output_adversarial_batch_id']
      img_count_adv_batch = len(self.adv_batches.data[adv_batch_id]['images'])
      if (img_count_adv_batch < 100) and (work['elapsed_time'] < 500):
        affected_adversarial_batches.add(adv_batch_id)
        new_attack_work[work_id] = dict(work)
        new_attack_work[work_id].update(
            {
                'claimed_worker_id': None,
                'claimed_worker_start_time': None,
                'is_completed': False,
                'error': None,
                'elapsed_time': None,
            })
    self.attack_work.replace_work(new_attack_work)
    print_header('Changes in attack works:')
    print(self.attack_work)
    # build list of classification batches
    self.class_batches.init_from_datastore()
    affected_class_batches = set()
    for k, v in iteritems(self.class_batches.data):
      if v['adversarial_batch_id'] in affected_adversarial_batches:
        affected_class_batches.add(k)
    # cleanup defense work on affected batches
    self.defense_work.read_all_from_datastore()
    new_defense_work = {}
    for k, v in iteritems(self.defense_work.work):
      if v['output_classification_batch_id'] in affected_class_batches:
        new_defense_work[k] = dict(v)
        new_defense_work[k].update(
            {
                'claimed_worker_id': None,
                'claimed_worker_start_time': None,
                'is_completed': False,
                'error': None,
                'elapsed_time': None,
                'stat_correct': None,
                'stat_error': None,
                'stat_target_class': None,
                'stat_num_images': None,
            })
    self.defense_work.replace_work(new_defense_work)
    print_header('Changes in defense works:')
    print(self.defense_work)
    print('')
    print('Total number of affected attack work: ', len(self.attack_work))
    print('Total number of affected defense work: ', len(self.defense_work))
    inp = input_str('Are you sure? (type "yes" without quotes to confirm): ')
    if inp != 'yes':
      return
    print('Writing attacks work')
    self.attack_work.write_all_to_datastore()
    print('Writing defenses work')
    self.defense_work.write_all_to_datastore()
    print('Done!')

  def _cleanup_keys_with_confirmation(self, keys_to_delete):
    """Asks confirmation and then deletes entries with keys.

    Args:
      keys_to_delete: list of datastore keys for which entries should be deleted
    """
    print('Round name: ', self.round_name)
    print('Number of entities to be deleted: ', len(keys_to_delete))
    if not keys_to_delete:
      return
    if self.verbose:
      print('Entities to delete:')
      idx = 0
      prev_key_prefix = None
      dots_printed_after_same_prefix = False
      for k in keys_to_delete:
        if idx >= 20:
          print('   ...')
          print('   ...')
          break
        key_prefix = (k.flat_path[0:1]
                      if k.flat_path[0] in [u'SubmissionType', u'WorkType']
                      else k.flat_path[0])
        if prev_key_prefix == key_prefix:
          if not dots_printed_after_same_prefix:
            print('   ...')
          dots_printed_after_same_prefix = True
        else:
          print('  ', k)
          dots_printed_after_same_prefix = False
          idx += 1
        prev_key_prefix = key_prefix
    print()
    inp = input_str('Are you sure? (type "yes" without quotes to confirm): ')
    if inp != 'yes':
      return
    with self.datastore_client.no_transact_batch() as batch:
      for k in keys_to_delete:
        batch.delete(k)
    print('Data deleted')

  def cleanup_defenses(self):
    """Cleans up all data about defense work in current round."""
    print_header('CLEANING UP DEFENSES DATA')
    work_ancestor_key = self.datastore_client.key('WorkType', 'AllDefenses')
    keys_to_delete = [
        e.key
        for e in self.datastore_client.query_fetch(kind=u'ClassificationBatch')
    ] + [
        e.key
        for e in self.datastore_client.query_fetch(kind=u'Work',
                                                   ancestor=work_ancestor_key)
    ]
    self._cleanup_keys_with_confirmation(keys_to_delete)

  def cleanup_datastore(self):
    """Cleans up datastore and deletes all information about current round."""
    print_header('CLEANING UP ENTIRE DATASTORE')
    kinds_to_delete = [u'Submission', u'SubmissionType',
                       u'DatasetImage', u'DatasetBatch',
                       u'AdversarialImage', u'AdversarialBatch',
                       u'Work', u'WorkType',
                       u'ClassificationBatch']
    keys_to_delete = [e.key for k in kinds_to_delete
                      for e in self.datastore_client.query_fetch(kind=k)]
    self._cleanup_keys_with_confirmation(keys_to_delete)


USAGE = """Use one of the following commands to run master:
  run_master.sh attack
  run_master.sh defense
  run_master.sh cleanup_defenses
  run_master.sh results
  run_master.sh status
  run_master.sh cleanup_datastore
"""


def main(args):
  """Main function which runs master."""
  if args.blacklisted_submissions:
    logging.warning('BLACKLISTED SUBMISSIONS: %s',
                    args.blacklisted_submissions)
  if args.limited_dataset:
    logging.info('Using limited dataset: 3 batches * 10 images')
    max_dataset_num_images = 30
    batch_size = 10
  else:
    logging.info('Using full dataset. Batch size: %d', DEFAULT_BATCH_SIZE)
    max_dataset_num_images = None
    batch_size = DEFAULT_BATCH_SIZE
  random.seed()
  print('\nRound: {0}\n'.format(args.round_name))
  eval_master = EvaluationMaster(
      storage_client=eval_lib.CompetitionStorageClient(
          args.project_id, args.storage_bucket),
      datastore_client=eval_lib.CompetitionDatastoreClient(
          args.project_id, args.round_name),
      round_name=args.round_name,
      dataset_name=args.dataset_name,
      blacklisted_submissions=args.blacklisted_submissions,
      results_dir=args.results_dir,
      num_defense_shards=args.num_defense_shards,
      verbose=args.verbose,
      batch_size=batch_size,
      max_dataset_num_images=max_dataset_num_images)
  if args.command == 'attack':
    eval_master.prepare_attacks()
  elif args.command == 'defense':
    eval_master.prepare_defenses()
  elif args.command == 'cleanup_defenses':
    eval_master.cleanup_defenses()
  elif args.command == 'results':
    eval_master.compute_results()
  elif args.command == 'status':
    eval_master.show_status()
  elif args.command == 'cleanup_datastore':
    eval_master.cleanup_datastore()
  elif args.command == 'cleanup_failed_attacks':
    eval_master.cleanup_failed_attacks()
  elif args.command == 'cleanup_attacks_with_zero_images':
    eval_master.cleanup_attacks_with_zero_images()
  else:
    print('Invalid command: ', args.command)
    print('')
    print(USAGE)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Master which coordinates all workers.')
  parser.add_argument('command',
                      help='Command to run. Possible commands include '
                           '"attack", "defense", "scores", "status".')
  parser.add_argument('--project_id',
                      required=True,
                      help='Your Google Cloud project ID.')
  parser.add_argument('--storage_bucket',
                      required=True,
                      help='Cloud Storage bucket to store competition data.')
  parser.add_argument('--round_name',
                      default='testing-round',
                      required=False,
                      help='Name of the round.')
  parser.add_argument('--dataset_name',
                      default='dev',
                      required=False,
                      help='Which dataset to use, either dev or final.')
  parser.add_argument('--blacklisted_submissions',
                      default='',
                      required=False,
                      help='Comma separated list of blacklisted submission '
                           'IDs.')
  parser.add_argument('--results_dir',
                      required=True,
                      help='Directory where to save results.')
  parser.add_argument('--num_defense_shards',
                      default=10,
                      required=False,
                      help='Number of defense shards')
  parser.add_argument('--limited_dataset', dest='limited_dataset',
                      action='store_true')
  parser.add_argument('--nolimited_dataset', dest='limited_dataset',
                      action='store_false')
  parser.set_defaults(limited_dataset=False)
  parser.add_argument('--verbose', dest='verbose', action='store_true')
  parser.add_argument('--noverbose', dest='verbose', action='store_false')
  parser.set_defaults(verbose=False)
  parser.add_argument('--log_file',
                      default='',
                      required=False,
                      help='Location of the logfile.')
  master_args = parser.parse_args()
  logging_args = {
      'format':
      '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s -- %(message)s',
      'level': logging.INFO,
      'datefmt': '%Y-%m-%d %H:%M:%S',
  }
  if master_args.log_file:
    logging_args['filename'] = master_args.log_file
    logging_args['filemode'] = 'w'
  logging.basicConfig(**logging_args)
  main(master_args)
