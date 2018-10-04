r"""Tool to validate submission for adversarial competition.

Usage:
  python validate_submission.py \
    --submission_filename=FILENAME \
    --submission_type=TYPE \
    [--use_gpu]

Where:
  FILENAME - filename of the submission
  TYPE - type of the submission, one of the following without quotes:
    "attack", "targeted_attack" or "defense"
  --use_gpu - if argument specified then submission will be run on GPU using
    nvidia-docker, otherwise will be run on CPU.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import random
import subprocess
import tempfile
import submission_validator_lib


def print_in_box(text):
  """
  Prints `text` surrounded by a box made of *s
  """
  print('')
  print('*' * (len(text) + 6))
  print('** ' + text + ' **')
  print('*' * (len(text) + 6))
  print('')


def main(args):
  """
  Validates the submission.
  """
  print_in_box('Validating submission ' + args.submission_filename)
  random.seed()
  temp_dir = args.temp_dir
  delete_temp_dir = False
  if not temp_dir:
    temp_dir = tempfile.mkdtemp()
    logging.info('Created temporary directory: %s', temp_dir)
    delete_temp_dir = True
  validator = submission_validator_lib.SubmissionValidator(temp_dir,
                                                           args.use_gpu)
  if validator.validate_submission(args.submission_filename,
                                   args.submission_type):
    print_in_box('Submission is VALID!')
  else:
    print_in_box('Submission is INVALID, see log messages for details')
  if delete_temp_dir:
    logging.info('Deleting temporary directory: %s', temp_dir)
    subprocess.call(['rm', '-rf', temp_dir])


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Submission validation script.')
  parser.add_argument('--submission_filename',
                      required=True,
                      help='Filename of the submission.')
  parser.add_argument('--submission_type',
                      required=True,
                      help='Type of the submission, '
                           'one of "attack", "targeted_attack" or "defense"')
  parser.add_argument('--temp_dir',
                      required=False,
                      default='',
                      help='Temporary directory to extract and run submission. '
                           'If empty then temporary directory will be created '
                           'by the script and then deleted in the end.')
  parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
  parser.add_argument('--nouse_gpu', dest='use_gpu', action='store_false')
  parser.set_defaults(use_gpu=False)
  loggint_format = ('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s -- '
                    '%(message)s')
  logging.basicConfig(format=loggint_format,
                      level=logging.INFO,
                      datefmt='%Y-%m-%d %H:%M:%S')
  main(parser.parse_args())
