r"""Tool to validate all submission and copy them to proper location.

Usage:
  python validate_and_copy_submissions.py \
    --source_dir=SOURCE \
    --target_dir=TARGET \
    [--containers_file=CONTAINER_FILE] \
    [--log_file=LOG_FILE] \
    [--nouse_gpu] \
    [--nocopy]

Where:
  SOURCE - Google Cloud Storage directory with all submissions to verify.
    Submissions in the source directory could be structured any way, this tool
    will go though all subdirectories and look for zip, tar and tar.gz archives.
  TARGET - Target directory in Google Cloud Storage, typically it should be
    gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/${ROUND_NAME}/submissions
  CONTAINER_FILE - optional name of the file where to save list of all Docker
    containers used by all submissions.
  LOG_FILE - optional filename of the logfile.
  --nouse_gpu - if argument is provided then submission will be run on CPU,
    otherwise will be run on GPU.
  --nocopy - if argument is provided then submissions will be validated,
    but no copy will be performed.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import logging
import os
import random
import shutil
import subprocess
import tempfile
from six import iteritems
import validate_submission_lib


ALLOWED_EXTENSIONS = ['.zip', '.tar', '.tar.gz']

TYPE_ATTACK = 'attack'
TYPE_TARGETED_ATTACK = 'targeted_attack'
TYPE_DEFENSE = 'defense'

TYPE_TO_DIR = {
    'attack': 'nontargeted',
    'targeted_attack': 'targeted',
    'defense': 'defense',
}


class ValidationStats(object):
  """Class which stores statistics about validation of the submissions."""

  def __init__(self):
    # dictionary mapping submission_type to tuple (num_success, num_fail)
    self.stats = {}

  def _update_stat(self, submission_type, increase_success, increase_fail):
    """Common method to update submission statistics."""
    stat = self.stats.get(submission_type, (0, 0))
    stat = (stat[0] + increase_success, stat[1] + increase_fail)
    self.stats[submission_type] = stat

  def add_success(self, submission_type):
    """Add one successfull submission of given type."""
    self._update_stat(submission_type, 1, 0)

  def add_failure(self, submission_type='unknown'):
    """Add one failed submission of given type."""
    self._update_stat(submission_type, 0, 1)

  def log_stats(self):
    """Print statistics into log."""
    logging.info('Validation statistics: ')
    for k, v in iteritems(self.stats):
      logging.info('%s - %d valid out of %d total submissions',
                   k, v[0], v[0] + v[1])


class SubmissionValidator(object):
  """Helper class which performs validation of all submissions."""

  def __init__(self, source_dir, target_dir, temp_dir, do_copy, use_gpu,
               containers_file=None):
    """Initializes SubmissionValidator.

    Args:
      source_dir: source Google Cloud Storage directory with all submissions
      target_dir: target Google Cloud Storage directory where to copy
        submissions
      temp_dir: local temporary directory
      do_copy: if True then validate and copy submissions, if False then only
        validate
      use_gpu: if True then use GPU for validation, otherwise use CPU
      containers_file: optional name of the local text file where list of
        Docker containes of all submissions will be saved.
    """
    self.source_dir = source_dir
    self.target_dir = target_dir
    self.do_copy = do_copy
    self.containers_file = containers_file
    self.list_of_containers = set()
    self.local_id_to_path_mapping_file = os.path.join(temp_dir,
                                                      'id_to_path_mapping.csv')
    self.validate_dir = os.path.join(temp_dir, 'validate')
    self.base_validator = validate_submission_lib.SubmissionValidator(
        self.validate_dir, use_gpu)
    self.stats = ValidationStats()
    self.download_dir = os.path.join(temp_dir, 'download')
    self.cur_submission_idx = 0
    self.id_to_path_mapping = {}

  def copy_submission_locally(self, cloud_path):
    """Copies submission from Google Cloud Storage to local directory.

    Args:
      cloud_path: path of the submission in Google Cloud Storage

    Returns:
      name of the local file where submission is copied to
    """
    local_path = os.path.join(self.download_dir, os.path.basename(cloud_path))
    cmd = ['gsutil', 'cp', cloud_path, local_path]
    if subprocess.call(cmd) != 0:
      logging.error('Can\'t copy submission locally')
      return None
    return local_path

  def copy_submission_to_destination(self, src_filename, dst_subdir,
                                     submission_id):
    """Copies submission to target directory.

    Args:
      src_filename: source filename of the submission
      dst_subdir: subdirectory of the target directory where submission should
        be copied to
      submission_id: ID of the submission, will be used as a new
        submission filename (before extension)
    """

    extension = [e for e in ALLOWED_EXTENSIONS if src_filename.endswith(e)]
    if len(extension) != 1:
      logging.error('Invalid submission extension: %s', src_filename)
      return
    dst_filename = os.path.join(self.target_dir, dst_subdir,
                                submission_id + extension[0])
    cmd = ['gsutil', 'cp', src_filename, dst_filename]
    if subprocess.call(cmd) != 0:
      logging.error('Can\'t copy submission to destination')
    else:
      logging.info('Submission copied to: %s', dst_filename)

  def validate_and_copy_one_submission(self, submission_path):
    """Validates one submission and copies it to target directory.

    Args:
      submission_path: path in Google Cloud Storage of the submission file
    """
    if os.path.exists(self.download_dir):
      shutil.rmtree(self.download_dir)
    os.makedirs(self.download_dir)
    if os.path.exists(self.validate_dir):
      shutil.rmtree(self.validate_dir)
    os.makedirs(self.validate_dir)
    logging.info('\n' + ('#' * 80) + '\n# Processing submission: %s\n'
                 + '#' * 80, submission_path)
    local_path = self.copy_submission_locally(submission_path)
    metadata = self.base_validator.validate_submission(local_path)
    if not metadata:
      logging.error('Submission "%s" is INVALID', submission_path)
      self.stats.add_failure()
      return
    submission_type = metadata['type']
    container_name = metadata['container_gpu']
    logging.info('Submission "%s" is VALID', submission_path)
    self.list_of_containers.add(container_name)
    self.stats.add_success(submission_type)
    if self.do_copy:
      submission_id = '{0:04}'.format(self.cur_submission_idx)
      self.cur_submission_idx += 1
      self.copy_submission_to_destination(submission_path,
                                          TYPE_TO_DIR[submission_type],
                                          submission_id)
      self.id_to_path_mapping[submission_id] = submission_path

  def save_id_to_path_mapping(self):
    """Saves mapping from submission IDs to original filenames.

    This mapping is saved as CSV file into target directory.
    """
    if not self.id_to_path_mapping:
      return
    with open(self.local_id_to_path_mapping_file, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['id', 'path'])
      for k, v in sorted(iteritems(self.id_to_path_mapping)):
        writer.writerow([k, v])
    cmd = ['gsutil', 'cp', self.local_id_to_path_mapping_file,
           os.path.join(self.target_dir, 'id_to_path_mapping.csv')]
    if subprocess.call(cmd) != 0:
      logging.error('Can\'t copy id_to_path_mapping.csv to target directory')

  def run(self):
    """Runs validation of all submissions."""
    cmd = ['gsutil', 'ls', os.path.join(self.source_dir, '**')]
    try:
      files_list = subprocess.check_output(cmd).split('\n')
    except subprocess.CalledProcessError:
      logging.error('Can''t read source directory')
    all_submissions = [
        s for s in files_list
        if s.endswith('.zip') or s.endswith('.tar') or s.endswith('.tar.gz')
    ]
    for submission_path in all_submissions:
      self.validate_and_copy_one_submission(submission_path)
    self.stats.log_stats()
    self.save_id_to_path_mapping()
    if self.containers_file:
      with open(self.containers_file, 'w') as f:
        f.write('\n'.join(sorted(self.list_of_containers)))


def main(args):
  """Validate all submissions and copy them into place"""
  random.seed()
  temp_dir = tempfile.mkdtemp()
  logging.info('Created temporary directory: %s', temp_dir)
  validator = SubmissionValidator(
      source_dir=args.source_dir,
      target_dir=args.target_dir,
      temp_dir=temp_dir,
      do_copy=args.copy,
      use_gpu=args.use_gpu,
      containers_file=args.containers_file)
  validator.run()
  logging.info('Deleting temporary directory: %s', temp_dir)
  subprocess.call(['rm', '-rf', temp_dir])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Submission validation script.')
  parser.add_argument('--source_dir',
                      required=True,
                      help='Source directory.')
  parser.add_argument('--target_dir',
                      required=True,
                      help='Target directory.')
  parser.add_argument('--log_file',
                      default='',
                      required=False,
                      help='Location of the logfile.')
  parser.add_argument('--containers_file',
                      default='',
                      required=False,
                      help='Local file with list of containers.')
  parser.add_argument('--copy', dest='copy', action='store_true')
  parser.add_argument('--nocopy', dest='copy', action='store_false')
  parser.set_defaults(copy=True)
  parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
  parser.add_argument('--nouse_gpu', dest='use_gpu', action='store_false')
  parser.set_defaults(use_gpu=True)
  command_line_args = parser.parse_args()
  logging_args = {
      'format': ('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s'
                 ' -- %(message)s'),
      'level': logging.INFO,
      'datefmt': '%Y-%m-%d %H:%M:%S',
  }
  if command_line_args.log_file:
    logging_args['filename'] = command_line_args.log_file
    logging_args['filemode'] = 'w'
  logging.basicConfig(**logging_args)
  main(command_line_args)
