"""Helper library which performs validation of the submission."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import logging
import os
import re
import subprocess

import numpy as np
from PIL import Image

from six import iteritems
from six import PY3


EXTRACT_COMMAND = {
    '.zip': ['unzip', '${src}', '-d', '${dst}'],
    '.tar': ['tar', 'xvf', '${src}', '-C', '${dst}'],
    '.tar.gz': ['tar', 'xvzf', '${src}', '-C', '${dst}'],
}

ALLOWED_SUBMISSION_TYPES = ['attack', 'targeted_attack', 'defense']

REQUIRED_METADATA_JSON_FIELDS = ['entry_point', 'container',
                                 'container_gpu', 'type']

CMD_VARIABLE_RE = re.compile('^\\$\\{(\\w+)\\}$')

BATCH_SIZE = 100
IMAGE_NAME_PATTERN = 'IMG{0:04}.png'

ALLOWED_EPS = [4, 8, 12, 16]

MAX_SUBMISSION_SIZE_ZIPPED = 8*1024*1024*1024      #  8 GiB
MAX_SUBMISSION_SIZE_UNPACKED = 16*1024*1024*1024   # 16 GiB
MAX_DOCKER_IMAGE_SIZE = 8*1024*1024*1024           #  8 GiB


def get_extract_command_template(filename):
  """Returns extraction command based on the filename extension."""
  for k, v in iteritems(EXTRACT_COMMAND):
    if filename.endswith(k):
      return v
  return None


def shell_call(command, **kwargs):
  """Calls shell command with parameter substitution.

  Args:
    command: command to run as a list of tokens
    **kwargs: dirctionary with substitutions

  Returns:
    whether command was successful, i.e. returned 0 status code

  Example of usage:
    shell_call(['cp', '${A}', '${B}'], A='src_file', B='dst_file')
  will call shell command:
    cp src_file dst_file
  """
  command = list(command)
  for i in range(len(command)):
    m = CMD_VARIABLE_RE.match(command[i])
    if m:
      var_id = m.group(1)
      if var_id in kwargs:
        command[i] = kwargs[var_id]
  return subprocess.call(command) == 0


def make_directory_writable(dirname):
  """Makes directory readable and writable by everybody.

  Args:
    dirname: name of the directory

  Returns:
    True if operation was successfull

  If you run something inside Docker container and it writes files, then
  these files will be written as root user with restricted permissions.
  So to be able to read/modify these files outside of Docker you have to change
  permissions to be world readable and writable.
  """
  retval = shell_call(['docker', 'run', '-v',
                       '{0}:/output_dir'.format(dirname),
                       'busybox:1.27.2',
                       'chmod', '-R', 'a+rwx', '/output_dir'])
  if not retval:
    logging.error('Failed to change permissions on directory: %s', dirname)
  return retval


def load_defense_output(filename):
  """Loads output of defense from given file."""
  result = {}
  with open(filename) as f:
    for row in csv.reader(f):
      try:
        image_filename = row[0]
        if not image_filename.endswith('.png'):
          image_filename += '.png'
        label = int(row[1])
      except (IndexError, ValueError):
        continue
      result[image_filename] = label
  return result


class SubmissionValidator(object):
  """Class which performs validation of the submission."""

  def __init__(self, temp_dir, use_gpu):
    """Initializes instance of SubmissionValidator.

    Args:
      temp_dir: temporary working directory
      use_gpu: whether to use GPU
    """
    self._temp_dir = temp_dir
    self._use_gpu = use_gpu
    self._tmp_extracted_dir = os.path.join(self._temp_dir, 'tmp_extracted')
    self._extracted_submission_dir = os.path.join(self._temp_dir, 'extracted')
    self._sample_input_dir = os.path.join(self._temp_dir, 'input')
    self._sample_output_dir = os.path.join(self._temp_dir, 'output')

  def _prepare_temp_dir(self):
    """Cleans up and prepare temporary directory."""
    shell_call(['rm', '-rf', os.path.join(self._temp_dir, '*')])
    # NOTE: we do not create self._extracted_submission_dir
    # this is intentional because self._tmp_extracted_dir or it's subdir
    # will be renames into self._extracted_submission_dir
    os.mkdir(self._tmp_extracted_dir)
    os.mkdir(self._sample_input_dir)
    os.mkdir(self._sample_output_dir)
    # make output dir world writable
    shell_call(['chmod', 'a+rwX', '-R', self._sample_output_dir])

  def _extract_submission(self, filename):
    """Extracts submission and moves it into self._extracted_submission_dir."""
    # verify filesize
    file_size = os.path.getsize(filename)
    if file_size > MAX_SUBMISSION_SIZE_ZIPPED:
      logging.error('Submission archive size %d is exceeding limit %d',
                    file_size, MAX_SUBMISSION_SIZE_ZIPPED)
      return False
    # determime archive type
    exctract_command_tmpl = get_extract_command_template(filename)
    if not exctract_command_tmpl:
      logging.error('Input file has to be zip, tar or tar.gz archive; however '
                    'found: %s', filename)
      return False
    # extract archive
    submission_dir = os.path.dirname(filename)
    submission_basename = os.path.basename(filename)
    logging.info('Extracting archive %s', filename)
    retval = shell_call(
        ['docker', 'run',
         '--network=none',
         '-v', '{0}:/input_dir'.format(submission_dir),
         '-v', '{0}:/output_dir'.format(self._tmp_extracted_dir),
         'busybox:1.27.2'] + exctract_command_tmpl,
        src=os.path.join('/input_dir', submission_basename),
        dst='/output_dir')
    if not retval:
      logging.error('Failed to extract submission from file %s', filename)
      return False
    if not make_directory_writable(self._tmp_extracted_dir):
      return False
    # find submission root
    root_dir = self._tmp_extracted_dir
    root_dir_content = [d for d in os.listdir(root_dir) if d != '__MACOSX']
    if (len(root_dir_content) == 1
        and os.path.isdir(os.path.join(root_dir, root_dir_content[0]))):
      logging.info('Looks like submission root is in subdirectory "%s" of '
                   'the archive', root_dir_content[0])
      root_dir = os.path.join(root_dir, root_dir_content[0])
    # Move files to self._extracted_submission_dir.
    # At this point self._extracted_submission_dir does not exist,
    # so following command will simply rename root_dir into
    # self._extracted_submission_dir
    if not shell_call(['mv', root_dir, self._extracted_submission_dir]):
      logging.error('Can''t move submission files from root directory')
      return False
    return True

  def _verify_submission_size(self):
    submission_size = 0
    for dirname, _, filenames in os.walk(self._extracted_submission_dir):
      for f in filenames:
        submission_size += os.path.getsize(os.path.join(dirname, f))
    logging.info('Unpacked submission size: %d', submission_size)
    if submission_size > MAX_SUBMISSION_SIZE_UNPACKED:
      logging.error('Submission size exceeding limit %d',
                    MAX_SUBMISSION_SIZE_UNPACKED)
    return submission_size <= MAX_SUBMISSION_SIZE_UNPACKED

  def _load_and_verify_metadata(self, submission_type):
    """Loads and verifies metadata.

    Args:
      submission_type: type of the submission

    Returns:
      dictionaty with metadata or None if metadata not found or invalid
    """
    metadata_filename = os.path.join(self._extracted_submission_dir,
                                     'metadata.json')
    if not os.path.isfile(metadata_filename):
      logging.error('metadata.json not found')
      return None
    try:
      with open(metadata_filename, 'r') as f:
        metadata = json.load(f)
    except IOError as e:
      logging.error('Failed to load metadata: %s', e)
      return None
    for field_name in REQUIRED_METADATA_JSON_FIELDS:
      if field_name not in metadata:
        logging.error('Field %s not found in metadata', field_name)
        return None
    # Verify submission type
    if submission_type != metadata['type']:
      logging.error('Invalid submission type in metadata, expected "%s", '
                    'actual "%s"', submission_type, metadata['type'])
      return None
    # Check submission entry point
    entry_point = metadata['entry_point']
    if not os.path.isfile(os.path.join(self._extracted_submission_dir,
                                       entry_point)):
      logging.error('Entry point not found: %s', entry_point)
      return None
    if not entry_point.endswith('.sh'):
      logging.warning('Entry point is not an .sh script. '
                      'This is not necessarily a problem, but if submission '
                      'won''t run double check entry point first: %s',
                      entry_point)
    # Metadata verified
    return metadata

  def _verify_docker_image_size(self, image_name):
    """Verifies size of Docker image.

    Args:
      image_name: name of the Docker image.

    Returns:
      True if image size is withing the limits, False otherwise.
    """
    shell_call(['docker', 'pull', image_name])
    try:
      image_size = subprocess.check_output(
          ['docker', 'inspect', '--format={{.Size}}', image_name]).strip()
      image_size = int(image_size) if PY3 else long(image_size)
    except (ValueError, subprocess.CalledProcessError) as e:
      logging.error('Failed to determine docker image size: %s', e)
      return False
    logging.info('Size of docker image %s is %d', image_name, image_size)
    if image_size > MAX_DOCKER_IMAGE_SIZE:
      logging.error('Image size exceeds limit %d', MAX_DOCKER_IMAGE_SIZE)
    return image_size <= MAX_DOCKER_IMAGE_SIZE

  def _prepare_sample_data(self, submission_type):
    """Prepares sample data for the submission.

    Args:
      submission_type: type of the submission.
    """
    # write images
    images = np.random.randint(0, 256,
                               size=[BATCH_SIZE, 299, 299, 3], dtype=np.uint8)
    for i in range(BATCH_SIZE):
      Image.fromarray(images[i, :, :, :]).save(
          os.path.join(self._sample_input_dir, IMAGE_NAME_PATTERN.format(i)))
    # write target class for targeted attacks
    if submission_type == 'targeted_attack':
      target_classes = np.random.randint(1, 1001, size=[BATCH_SIZE])
      target_class_filename = os.path.join(self._sample_input_dir,
                                           'target_class.csv')
      with open(target_class_filename, 'w') as f:
        for i in range(BATCH_SIZE):
          f.write((IMAGE_NAME_PATTERN + ',{1}\n').format(i, target_classes[i]))

  def _run_submission(self, metadata):
    """Runs submission inside Docker container.

    Args:
      metadata: dictionary with submission metadata

    Returns:
      True if status code of Docker command was success (i.e. zero),
      False otherwise.
    """
    if self._use_gpu:
      docker_binary = 'nvidia-docker'
      container_name = metadata['container_gpu']
    else:
      docker_binary = 'docker'
      container_name = metadata['container']
    if metadata['type'] == 'defense':
      cmd = [docker_binary, 'run',
             '--network=none',
             '-m=24g',
             '-v', '{0}:/input_images:ro'.format(self._sample_input_dir),
             '-v', '{0}:/output_data'.format(self._sample_output_dir),
             '-v', '{0}:/code'.format(self._extracted_submission_dir),
             '-w', '/code',
             container_name,
             './' + metadata['entry_point'],
             '/input_images',
             '/output_data/result.csv']
    else:
      epsilon = np.random.choice(ALLOWED_EPS)
      cmd = [docker_binary, 'run',
             '--network=none',
             '-m=24g',
             '-v', '{0}:/input_images:ro'.format(self._sample_input_dir),
             '-v', '{0}:/output_images'.format(self._sample_output_dir),
             '-v', '{0}:/code'.format(self._extracted_submission_dir),
             '-w', '/code',
             container_name,
             './' + metadata['entry_point'],
             '/input_images',
             '/output_images',
             str(epsilon)]
    logging.info('Command to run submission: %s', ' '.join(cmd))
    return shell_call(cmd)

  def _verify_output(self, submission_type):
    """Verifies correctness of the submission output.

    Args:
      submission_type: type of the submission

    Returns:
      True if output looks valid
    """
    result = True
    if submission_type == 'defense':
      try:
        image_classification = load_defense_output(
            os.path.join(self._sample_output_dir, 'result.csv'))
        expected_keys = [IMAGE_NAME_PATTERN.format(i)
                         for i in range(BATCH_SIZE)]
        if set(image_classification.keys()) != set(expected_keys):
          logging.error('Classification results are not saved for all images')
          result = False
      except IOError as e:
        logging.error('Failed to read defense output file: %s', e)
        result = False
    else:
      for i in range(BATCH_SIZE):
        image_filename = os.path.join(self._sample_output_dir,
                                      IMAGE_NAME_PATTERN.format(i))
        try:
          img = np.array(Image.open(image_filename).convert('RGB'))
          if list(img.shape) != [299, 299, 3]:
            logging.error('Invalid image size %s for image %s',
                          str(img.shape), image_filename)
            result = False
        except IOError as e:
          result = False
    return result

  def validate_submission(self, filename, submission_type):
    """Validates submission.

    Args:
      filename: submission filename
      submission_type: type of the submission,
        one of 'attack', 'targeted_attack' or 'defense'

    Returns:
      whether submission is valid
    """
    if submission_type not in ALLOWED_SUBMISSION_TYPES:
      logging.error('Invalid submission type: %s', submission_type)
      return False
    self._prepare_temp_dir()
    # Convert filename to be absolute path,
    # relative path might cause problems when monting directory in Docker
    filename = os.path.abspath(filename)
    # extract submission
    if not self._extract_submission(filename):
      return False
    # verify submission size
    if not self._verify_submission_size():
      return False
    # Load metadata
    metadata = self._load_and_verify_metadata(submission_type)
    if not metadata:
      return False
    # verify docker container size
    if not self._verify_docker_image_size(metadata['container_gpu']):
      return False
    # Try to run submission on sample data
    self._prepare_sample_data(submission_type)
    if not self._run_submission(metadata):
      logging.error('Failure while running submission')
      return False
    if not self._verify_output(submission_type):
      logging.warning('Some of the outputs of your submission are invalid or '
                      'missing. You submission still will be evaluation '
                      'but you might get lower score.')
    return True
