"""Worker which runs all computations on Cloud VMs.

Evaluation of competition is split into work pieces. One work piece is a
either evaluation of an attack on a batch of images or evaluation of a
defense on a batch of adversarial images.
All pieces of attack work are independent from each other and could be run
in parallel. Same for pieces of defense work - they are independent from each
other and could be run in parallel. But defense work could be run only after
all attack work is completed.

Worker first runs all attack pieces, by querying next piece of undone work
and running it. After all attack pieces are done, worker runs all defense pieces
in a similar way.

Before workers could be started, datastore has to be populated by master
with description of work to be done. See master.py for details.

NOTE: Worker is designed to run on linux machine with NVidia docker
installed. Worker generally needs administrative privilege to run properly.
Also worker relies on very specific directory structure created in home
directory. That's why it's highly recommended to run worker only in VM.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import int # long in python 2

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import time
import uuid

from six import iteritems

import eval_lib

from cleverhans.utils import shell_call


# Sleep time while waiting for next available piece of work
SLEEP_TIME = 30
SLEEP_TIME_SHORT = 10

# Time limit to run one pice of work
SUBMISSION_TIME_LIMIT = 500

# Set of local temporary directories and files
LOCAL_EVAL_ROOT_DIR = os.path.expanduser('~/competition_eval')
LOCAL_DATASET_DIR = os.path.expanduser('~/competition_eval/dataset_images')
LOCAL_SUBMISSIONS_DIR = os.path.expanduser('~/competition_eval/submissions')
LOCAL_INPUT_DIR = os.path.expanduser('~/competition_eval/input')
LOCAL_OUTPUT_DIR = os.path.expanduser('~/competition_eval/output')
LOCAL_PROCESSED_OUTPUT_DIR = os.path.expanduser(
    '~/competition_eval/processed_output')
LOCAL_ZIPPED_OUTPUT_DIR = os.path.expanduser(
    '~/competition_eval/zipped_output')
LOCAL_DATASET_METADATA_FILE = os.path.expanduser(
    '~/competition_eval/dataset_meta.csv')
LOCAL_DATASET_COPY = os.path.expanduser('~/competition_data/dataset')

# Types of submissions
TYPE_TARGETED = 'targeted'
TYPE_NONTARGETED = 'nontargeted'
TYPE_DEFENSE = 'defense'

# Extraction commands for various types of archive
EXTRACT_COMMAND = {
    '.zip': ['unzip', '${src}', '-d', '${dst}'],
    '.tar': ['tar', 'xvf', '${src}', '-C', '${dst}'],
    '.tar.gz': ['tar', 'xvzf', '${src}', '-C', '${dst}'],
}

# Docker binary to use
DOCKER_BINARY = 'docker'
DOCKER_NVIDIA_RUNTIME = '--runtime=nvidia'

# Names of relevant fields in submission metadata file
METADATA_CONTAINER = 'container_gpu'
METADATA_ENTRY_POINT = 'entry_point'
METADATA_TYPE = 'type'

# Mapping from submission type in metadata to submission type used in worker
METADATA_JSON_TYPE_TO_TYPE = {
    'attack': TYPE_NONTARGETED,
    'targeted_attack': TYPE_TARGETED,
    'defense': TYPE_DEFENSE,
}


def make_directory_writable(dirname):
  """Makes directory readable and writable by everybody.

  If you run something inside Docker container and it writes files, then
  these files will be written as root user with restricted permissions.
  So to be able to read/modify these files outside of Docker you have to change
  permissions to be world readable and writable.

  Args:
    dirname: name of the directory

  Returns:
    True if operation was successfull
  """
  shell_call(['docker', 'run', '-v',
              '{0}:/output_dir'.format(dirname),
              'busybox:1.27.2',
              'chmod', '-R', 'a+rwx', '/output_dir'])


def sudo_remove_dirtree(dir_name):
  """Removes directory tree as a superuser.

  Args:
    dir_name: name of the directory to remove.

  This function is necessary to cleanup directories created from inside a
  Docker, since they usually written as a root, thus have to be removed as a
  root.
  """
  try:
    subprocess.check_output(['sudo', 'rm', '-rf', dir_name])
  except subprocess.CalledProcessError as e:
    raise WorkerError('Can''t remove directory {0}'.format(dir_name), e)


class WorkerError(Exception):
  """Error which happen during evaluation of submission.

  To simplify error handling, worker only raises this type of exception.
  Exceptions of different types raised by other modules encapsulated
  into WorkerError by the worker.
  """

  def __init__(self, message, exc=None):
    """Initializes WorkerError.

    Args:
      message: error message
      exc: optional underlying exception.
    """
    super(WorkerError, self).__init__()
    self.msg = message
    self.exc = exc

  def __str__(self):
    """Returns human readable string representation of the exception."""
    if self.exc:
      return '{0}\nUnderlying exception:\n{1}'.format(self.msg, self.exc)
    else:
      return self.msg


def get_id_of_running_docker(container_name):
  """Returns ID of running docker container."""
  return shell_call([DOCKER_BINARY,
                     'ps',
                     '-q',
                     '--filter=name={}'.format(container_name)]).strip()


def is_docker_still_running(container_name):
  """Returns whether given Docker container is still running."""
  return bool(get_id_of_running_docker(container_name))


def kill_docker_container(container_name):
  """Kills given docker container."""
  docker_id = get_id_of_running_docker(container_name)
  shell_call([DOCKER_BINARY, 'stop', docker_id])


class ExecutableSubmission(object):
  """Base class which is used to run submissions."""

  def __init__(self, submission_id, submissions, storage_bucket):
    """Initializes ExecutableSubmission.

    Args:
      submission_id: ID of the submissions
      submissions: instance of CompetitionSubmissions with all submissions
      storage_bucket: storage bucket where all submissions are stored

    Raises:
      WorkerError: if submission was not found
    """
    self.submission_id = submission_id
    self.storage_bucket = storage_bucket
    self.type = None
    self.submission = None
    if submission_id in submissions.attacks:
      self.type = TYPE_NONTARGETED
      self.submission = submissions.attacks[submission_id]
    elif submission_id in submissions.targeted_attacks:
      self.type = TYPE_TARGETED
      self.submission = submissions.targeted_attacks[submission_id]
    elif submission_id in submissions.defenses:
      self.type = TYPE_DEFENSE
      self.submission = submissions.defenses[submission_id]
    else:
      raise WorkerError(
          'Submission with ID "{0}" not found'.format(submission_id))
    self.submission_dir = None
    self.extracted_submission_dir = None

  def download(self):
    """Method which downloads submission to local directory."""
    # Structure of the download directory:
    # submission_dir=LOCAL_SUBMISSIONS_DIR/submission_id
    # submission_dir/s.ext   <-- archived submission
    # submission_dir/extracted      <-- extracted submission

    # Check whether submission is already there
    if self.extracted_submission_dir:
      return
    self.submission_dir = os.path.join(LOCAL_SUBMISSIONS_DIR,
                                       self.submission_id)
    if (os.path.isdir(self.submission_dir)
        and os.path.isdir(os.path.join(self.submission_dir, 'extracted'))):
      # submission already there, just re-read metadata
      self.extracted_submission_dir = os.path.join(self.submission_dir,
                                                   'extracted')
      with open(os.path.join(self.extracted_submission_dir, 'metadata.json'),
                'r') as f:
        meta_json = json.load(f)
      self.container_name = str(meta_json[METADATA_CONTAINER])
      self.entry_point = str(meta_json[METADATA_ENTRY_POINT])
      return
    # figure out submission location in the Cloud and determine extractor
    submission_cloud_path = os.path.join('gs://', self.storage_bucket,
                                         self.submission.path)
    extract_command_tmpl = None
    extension = None
    for k, v in iteritems(EXTRACT_COMMAND):
      if submission_cloud_path.endswith(k):
        extension = k
        extract_command_tmpl = v
        break
    if not extract_command_tmpl:
      raise WorkerError('Unsupported submission extension')
    # download archive
    try:
      os.makedirs(self.submission_dir)
      tmp_extract_dir = os.path.join(self.submission_dir, 'tmp')
      os.makedirs(tmp_extract_dir)
      download_path = os.path.join(self.submission_dir, 's' + extension)
      try:
        logging.info('Downloading submission from %s to %s',
                     submission_cloud_path, download_path)
        shell_call(['gsutil', 'cp', submission_cloud_path, download_path])
      except subprocess.CalledProcessError as e:
        raise WorkerError('Can''t copy submission locally', e)
      # extract archive
      try:
        shell_call(extract_command_tmpl,
                   src=download_path, dst=tmp_extract_dir)
      except subprocess.CalledProcessError as e:
        # proceed even if extraction returned non zero error code,
        # sometimes it's just warning
        logging.warning('Submission extraction returned non-zero error code. '
                        'It may be just a warning, continuing execution. '
                        'Error: %s', e)
      try:
        make_directory_writable(tmp_extract_dir)
      except subprocess.CalledProcessError as e:
        raise WorkerError('Can''t make submission directory writable', e)
      # determine root of the submission
      tmp_root_dir = tmp_extract_dir
      root_dir_content = [d for d in os.listdir(tmp_root_dir)
                          if d != '__MACOSX']
      if (len(root_dir_content) == 1
          and os.path.isdir(os.path.join(tmp_root_dir, root_dir_content[0]))):
        tmp_root_dir = os.path.join(tmp_root_dir, root_dir_content[0])
      # move files to extract subdirectory
      self.extracted_submission_dir = os.path.join(self.submission_dir,
                                                   'extracted')
      try:
        shell_call(['mv', os.path.join(tmp_root_dir),
                    self.extracted_submission_dir])
      except subprocess.CalledProcessError as e:
        raise WorkerError('Can''t move submission files', e)
      # read metadata file
      try:
        with open(os.path.join(self.extracted_submission_dir, 'metadata.json'),
                  'r') as f:
          meta_json = json.load(f)
      except IOError as e:
        raise WorkerError(
            'Can''t read metadata.json for submission "{0}"'.format(
                self.submission_id),
            e)
      try:
        self.container_name = str(meta_json[METADATA_CONTAINER])
        self.entry_point = str(meta_json[METADATA_ENTRY_POINT])
        type_from_meta = METADATA_JSON_TYPE_TO_TYPE[meta_json[METADATA_TYPE]]
      except KeyError as e:
        raise WorkerError('Invalid metadata.json file', e)
      if type_from_meta != self.type:
        raise WorkerError('Inconsistent submission type in metadata: '
                          + type_from_meta + ' vs ' + self.type)
    except WorkerError as e:
      self.extracted_submission_dir = None
      sudo_remove_dirtree(self.submission_dir)
      raise

  def temp_copy_extracted_submission(self):
    """Creates a temporary copy of extracted submission.

    When executed, submission is allowed to modify it's own directory. So
    to ensure that submission does not pass any data between runs, new
    copy of the submission is made before each run. After a run temporary copy
    of submission is deleted.

    Returns:
      directory where temporary copy is located
    """
    tmp_copy_dir = os.path.join(self.submission_dir, 'tmp_copy')
    shell_call(['cp', '-R', os.path.join(self.extracted_submission_dir),
                tmp_copy_dir])
    return tmp_copy_dir

  def run_without_time_limit(self, cmd):
    """Runs docker command without time limit.

    Args:
      cmd: list with the command line arguments which are passed to docker
        binary

    Returns:
      how long it took to run submission in seconds

    Raises:
      WorkerError: if error occurred during execution of the submission
    """
    cmd = [DOCKER_BINARY, 'run', DOCKER_NVIDIA_RUNTIME] + cmd
    logging.info('Docker command: %s', ' '.join(cmd))
    start_time = time.time()
    retval = subprocess.call(cmd)
    elapsed_time_sec = int(time.time() - start_time)
    logging.info('Elapsed time of attack: %d', elapsed_time_sec)
    logging.info('Docker retval: %d', retval)
    if retval != 0:
      logging.warning('Docker returned non-zero retval: %d', retval)
      raise WorkerError('Docker returned non-zero retval ' + str(retval))
    return elapsed_time_sec

  def run_with_time_limit(self, cmd, time_limit=SUBMISSION_TIME_LIMIT):
    """Runs docker command and enforces time limit.

    Args:
      cmd: list with the command line arguments which are passed to docker
        binary after run
      time_limit: time limit, in seconds. Negative value means no limit.

    Returns:
      how long it took to run submission in seconds

    Raises:
      WorkerError: if error occurred during execution of the submission
    """
    if time_limit < 0:
      return self.run_without_time_limit(cmd)
    container_name = str(uuid.uuid4())
    cmd = [DOCKER_BINARY, 'run', DOCKER_NVIDIA_RUNTIME,
           '--detach', '--name', container_name] + cmd
    logging.info('Docker command: %s', ' '.join(cmd))
    logging.info('Time limit %d seconds', time_limit)
    retval = subprocess.call(cmd)
    start_time = time.time()
    elapsed_time_sec = 0
    while is_docker_still_running(container_name):
      elapsed_time_sec = int(time.time() - start_time)
      if elapsed_time_sec < time_limit:
        time.sleep(1)
      else:
        kill_docker_container(container_name)
        logging.warning('Submission was killed because run out of time')
    logging.info('Elapsed time of submission: %d', elapsed_time_sec)
    logging.info('Docker retval: %d', retval)
    if retval != 0:
      logging.warning('Docker returned non-zero retval: %d', retval)
      raise WorkerError('Docker returned non-zero retval ' + str(retval))
    return elapsed_time_sec


class AttackSubmission(ExecutableSubmission):
  """Class to run attack submissions."""

  def __init__(self, submission_id, submissions, storage_bucket):
    """Initializes AttackSubmission.

    Args:
      submission_id: ID of the submission
      submissions: instance of CompetitionSubmissions with all submissions
      storage_bucket: storage bucket where all submissions are stored

    Raises:
      WorkerError: if submission has incorrect type
    """
    super(AttackSubmission, self).__init__(submission_id, submissions,
                                           storage_bucket)
    if (self.type != TYPE_TARGETED) and (self.type != TYPE_NONTARGETED):
      raise WorkerError('Incorrect attack type for submission "{0}"'.format(
          submission_id))

  def run(self, input_dir, output_dir, epsilon):
    """Runs attack inside Docker.

    Args:
      input_dir: directory with input (dataset).
      output_dir: directory where output (adversarial images) should be written.
      epsilon: maximum allowed size of adversarial perturbation,
        should be in range [0, 255].

    Returns:
      how long it took to run submission in seconds
    """
    logging.info('Running attack %s', self.submission_id)
    tmp_run_dir = self.temp_copy_extracted_submission()
    cmd = ['--network=none',
           '-m=24g',
           '--cpus=3.75',
           '-v', '{0}:/input_images:ro'.format(input_dir),
           '-v', '{0}:/output_images'.format(output_dir),
           '-v', '{0}:/code'.format(tmp_run_dir),
           '-w', '/code',
           self.container_name,
           './' + self.entry_point,
           '/input_images',
           '/output_images',
           str(epsilon)]
    elapsed_time_sec = self.run_with_time_limit(cmd)
    sudo_remove_dirtree(tmp_run_dir)
    return elapsed_time_sec


class DefenseSubmission(ExecutableSubmission):
  """Helper class to run one defense submission."""

  def __init__(self, submission_id, submissions, storage_bucket):
    """Initializes DefenseSubmission.

    Args:
      submission_id: ID of the submission
      submissions: instance of CompetitionSubmissions with all submissions
      storage_bucket: storage bucket where all submissions are stored

    Raises:
      WorkerError: if submission has incorrect type
    """
    super(DefenseSubmission, self).__init__(submission_id, submissions,
                                            storage_bucket)
    if self.type != TYPE_DEFENSE:
      raise WorkerError('Incorrect defense type for submission "{0}"'.format(
          submission_id))

  def run(self, input_dir, output_file_path):
    """Runs defense inside Docker.

    Args:
      input_dir: directory with input (adversarial images).
      output_file_path: path of the output file.

    Returns:
      how long it took to run submission in seconds
    """
    logging.info('Running defense %s', self.submission_id)
    tmp_run_dir = self.temp_copy_extracted_submission()
    output_dir = os.path.dirname(output_file_path)
    output_filename = os.path.basename(output_file_path)
    cmd = ['--network=none',
           '-m=24g',
           '--cpus=3.75',
           '-v', '{0}:/input_images:ro'.format(input_dir),
           '-v', '{0}:/output_data'.format(output_dir),
           '-v', '{0}:/code'.format(tmp_run_dir),
           '-w', '/code',
           self.container_name,
           './' + self.entry_point,
           '/input_images',
           '/output_data/' + output_filename]
    elapsed_time_sec = self.run_with_time_limit(cmd)
    sudo_remove_dirtree(tmp_run_dir)
    return elapsed_time_sec


class EvaluationWorker(object):
  """Class which encapsulate logit of the worker.

  Main entry point of this class is EvaluationWorker.run_work method which
  performs cleanup of temporary directories, then runs
  EvaluationWorker.run_attacks and EvaluationWorker.run_defenses
  """

  def __init__(self, worker_id, storage_client, datastore_client,
               storage_bucket, round_name, dataset_name,
               blacklisted_submissions='', num_defense_shards=None):
    """Initializes EvaluationWorker.

    Args:
      worker_id: ID of the worker
      storage_client: instance of eval_lib.CompetitionStorageClient
      datastore_client: instance of eval_lib.CompetitionDatastoreClient
      storage_bucket: name of the Google Cloud Storage bucket where all
        competition data is stored
      round_name: name of the competition round
      dataset_name: name of the dataset to use, typically 'dev' of 'final'
      blacklisted_submissions: optional list of blacklisted submissions which
        are excluded from evaluation
      num_defense_shards: optional number of shards to use for evaluation of
        defenses
    """
    self.worker_id = int(worker_id)
    self.storage_client = storage_client
    self.datastore_client = datastore_client
    self.storage_bucket = storage_bucket
    self.round_name = round_name
    self.dataset_name = dataset_name
    self.blacklisted_submissions = [s.strip()
                                    for s in blacklisted_submissions.split(',')]
    if num_defense_shards:
      self.num_defense_shards = int(num_defense_shards)
    else:
      self.num_defense_shards = None
    logging.info('Number of defense shards: %s', str(self.num_defense_shards))
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
    self.attack_work = eval_lib.AttackWorkPieces(
        datastore_client=self.datastore_client)
    self.defense_work = eval_lib.DefenseWorkPieces(
        datastore_client=self.datastore_client)
    self.class_batches = eval_lib.ClassificationBatches(
        datastore_client=self.datastore_client,
        storage_client=self.storage_client,
        round_name=self.round_name)
    # whether data was initialized
    self.attacks_data_initialized = False
    self.defenses_data_initialized = False
    # dataset metadata
    self.dataset_meta = None

  def read_dataset_metadata(self):
    """Read `dataset_meta` field from bucket"""
    if self.dataset_meta:
      return
    shell_call(['gsutil', 'cp',
                'gs://' + self.storage_client.bucket_name + '/'
                + 'dataset/' + self.dataset_name + '_dataset.csv',
                LOCAL_DATASET_METADATA_FILE])
    with open(LOCAL_DATASET_METADATA_FILE, 'r') as f:
      self.dataset_meta = eval_lib.DatasetMetadata(f)

  def fetch_attacks_data(self):
    """Initializes data necessary to execute attacks.

    This method could be called multiple times, only first call does
    initialization, subsequent calls are noop.
    """
    if self.attacks_data_initialized:
      return
    # init data from datastore
    self.submissions.init_from_datastore()
    self.dataset_batches.init_from_datastore()
    self.adv_batches.init_from_datastore()
    # copy dataset locally
    if not os.path.exists(LOCAL_DATASET_DIR):
      os.makedirs(LOCAL_DATASET_DIR)
    eval_lib.download_dataset(self.storage_client, self.dataset_batches,
                              LOCAL_DATASET_DIR,
                              os.path.join(LOCAL_DATASET_COPY,
                                           self.dataset_name, 'images'))
    # download dataset metadata
    self.read_dataset_metadata()
    # mark as initialized
    self.attacks_data_initialized = True

  def run_attack_work(self, work_id):
    """Runs one attack work.

    Args:
      work_id: ID of the piece of work to run

    Returns:
      elapsed_time_sec, submission_id - elapsed time and id of the submission

    Raises:
      WorkerError: if error occurred during execution.
    """
    adv_batch_id = (
        self.attack_work.work[work_id]['output_adversarial_batch_id'])
    adv_batch = self.adv_batches[adv_batch_id]
    dataset_batch_id = adv_batch['dataset_batch_id']
    submission_id = adv_batch['submission_id']
    epsilon = self.dataset_batches[dataset_batch_id]['epsilon']
    logging.info('Attack work piece: '
                 'dataset_batch_id="%s" submission_id="%s" '
                 'epsilon=%d', dataset_batch_id, submission_id, epsilon)
    if submission_id in self.blacklisted_submissions:
      raise WorkerError('Blacklisted submission')
    # get attack
    attack = AttackSubmission(submission_id, self.submissions,
                              self.storage_bucket)
    attack.download()
    # prepare input
    input_dir = os.path.join(LOCAL_DATASET_DIR, dataset_batch_id)
    if attack.type == TYPE_TARGETED:
      # prepare file with target classes
      target_class_filename = os.path.join(input_dir, 'target_class.csv')
      self.dataset_meta.save_target_classes_for_batch(target_class_filename,
                                                      self.dataset_batches,
                                                      dataset_batch_id)
    # prepare output directory
    if os.path.exists(LOCAL_OUTPUT_DIR):
      sudo_remove_dirtree(LOCAL_OUTPUT_DIR)
    os.mkdir(LOCAL_OUTPUT_DIR)
    if os.path.exists(LOCAL_PROCESSED_OUTPUT_DIR):
      shutil.rmtree(LOCAL_PROCESSED_OUTPUT_DIR)
    os.mkdir(LOCAL_PROCESSED_OUTPUT_DIR)
    if os.path.exists(LOCAL_ZIPPED_OUTPUT_DIR):
      shutil.rmtree(LOCAL_ZIPPED_OUTPUT_DIR)
    os.mkdir(LOCAL_ZIPPED_OUTPUT_DIR)
    # run attack
    elapsed_time_sec = attack.run(input_dir, LOCAL_OUTPUT_DIR, epsilon)
    if attack.type == TYPE_TARGETED:
      # remove target class file
      os.remove(target_class_filename)
    # enforce epsilon and compute hashes
    image_hashes = eval_lib.enforce_epsilon_and_compute_hash(
        input_dir, LOCAL_OUTPUT_DIR, LOCAL_PROCESSED_OUTPUT_DIR, epsilon)
    if not image_hashes:
      logging.warning('No images saved by the attack.')
      return elapsed_time_sec, submission_id
    # write images back to datastore
    # rename images and add information to adversarial batch
    for clean_image_id, hash_val in iteritems(image_hashes):
      # we will use concatenation of batch_id and image_id
      # as adversarial image id and as a filename of adversarial images
      adv_img_id = adv_batch_id + '_' + clean_image_id
      # rename the image
      os.rename(
          os.path.join(LOCAL_PROCESSED_OUTPUT_DIR, clean_image_id + '.png'),
          os.path.join(LOCAL_PROCESSED_OUTPUT_DIR, adv_img_id + '.png'))
      # populate values which will be written to datastore
      image_path = '{0}/adversarial_images/{1}/{1}.zip/{2}.png'.format(
          self.round_name, adv_batch_id, adv_img_id)
      # u'' + foo is a a python 2/3 compatible way of casting foo to unicode
      adv_batch['images'][adv_img_id] = {
          'clean_image_id': u'' + str(clean_image_id),
          'image_path': u'' + str(image_path),
          'image_hash': u'' + str(hash_val),
      }
    # archive all images and copy to storage
    zipped_images_filename = os.path.join(LOCAL_ZIPPED_OUTPUT_DIR,
                                          adv_batch_id + '.zip')
    try:
      logging.debug('Compressing adversarial images to %s',
                    zipped_images_filename)
      shell_call([
          'zip', '-j', '-r', zipped_images_filename,
          LOCAL_PROCESSED_OUTPUT_DIR])
    except subprocess.CalledProcessError as e:
      raise WorkerError('Can''t make archive from adversarial iamges', e)
    # upload archive to storage
    dst_filename = '{0}/adversarial_images/{1}/{1}.zip'.format(
        self.round_name, adv_batch_id)
    logging.debug(
        'Copying archive with adversarial images to %s', dst_filename)
    self.storage_client.new_blob(dst_filename).upload_from_filename(
        zipped_images_filename)
    # writing adv batch to datastore
    logging.debug('Writing adversarial batch to datastore')
    self.adv_batches.write_single_batch_images_to_datastore(adv_batch_id)
    return elapsed_time_sec, submission_id

  def run_attacks(self):
    """Method which evaluates all attack work.

    In a loop this method queries not completed attack work, picks one
    attack work and runs it.
    """
    logging.info('******** Start evaluation of attacks ********')
    prev_submission_id = None
    while True:
      # wait until work is available
      self.attack_work.read_all_from_datastore()
      if not self.attack_work.work:
        logging.info('Work is not populated, waiting...')
        time.sleep(SLEEP_TIME)
        continue
      if self.attack_work.is_all_work_competed():
        logging.info('All attack work completed.')
        break
      # download all attacks data and dataset
      self.fetch_attacks_data()
      # pick piece of work
      work_id = self.attack_work.try_pick_piece_of_work(
          self.worker_id, submission_id=prev_submission_id)
      if not work_id:
        logging.info('Failed to pick work, waiting...')
        time.sleep(SLEEP_TIME_SHORT)
        continue
      logging.info('Selected work_id: %s', work_id)
      # execute work
      try:
        elapsed_time_sec, prev_submission_id = self.run_attack_work(work_id)
        logging.info('Work %s is done', work_id)
        # indicate that work is completed
        is_work_update = self.attack_work.update_work_as_completed(
            self.worker_id, work_id,
            other_values={'elapsed_time': elapsed_time_sec})
      except WorkerError as e:
        logging.info('Failed to run work:\n%s', str(e))
        is_work_update = self.attack_work.update_work_as_completed(
            self.worker_id, work_id, error=str(e))
      if not is_work_update:
        logging.warning('Can''t update work "%s" as completed by worker %d',
                        work_id, self.worker_id)
    logging.info('******** Finished evaluation of attacks ********')

  def fetch_defense_data(self):
    """Lazy initialization of data necessary to execute defenses."""
    if self.defenses_data_initialized:
      return
    logging.info('Fetching defense data from datastore')
    # init data from datastore
    self.submissions.init_from_datastore()
    self.dataset_batches.init_from_datastore()
    self.adv_batches.init_from_datastore()
    # read dataset metadata
    self.read_dataset_metadata()
    # mark as initialized
    self.defenses_data_initialized = True

  def run_defense_work(self, work_id):
    """Runs one defense work.

    Args:
      work_id: ID of the piece of work to run

    Returns:
      elapsed_time_sec, submission_id - elapsed time and id of the submission

    Raises:
      WorkerError: if error occurred during execution.
    """
    class_batch_id = (
        self.defense_work.work[work_id]['output_classification_batch_id'])
    class_batch = self.class_batches.read_batch_from_datastore(class_batch_id)
    adversarial_batch_id = class_batch['adversarial_batch_id']
    submission_id = class_batch['submission_id']
    cloud_result_path = class_batch['result_path']
    logging.info('Defense work piece: '
                 'adversarial_batch_id="%s" submission_id="%s"',
                 adversarial_batch_id, submission_id)
    if submission_id in self.blacklisted_submissions:
      raise WorkerError('Blacklisted submission')
    # get defense
    defense = DefenseSubmission(submission_id, self.submissions,
                                self.storage_bucket)
    defense.download()
    # prepare input - copy adversarial batch locally
    input_dir = os.path.join(LOCAL_INPUT_DIR, adversarial_batch_id)
    if os.path.exists(input_dir):
      sudo_remove_dirtree(input_dir)
    os.makedirs(input_dir)
    try:
      shell_call([
          'gsutil', '-m', 'cp',
          # typical location of adv batch:
          # testing-round/adversarial_images/ADVBATCH000/
          os.path.join('gs://', self.storage_bucket, self.round_name,
                       'adversarial_images', adversarial_batch_id, '*'),
          input_dir
      ])
      adv_images_files = os.listdir(input_dir)
      if (len(adv_images_files) == 1) and adv_images_files[0].endswith('.zip'):
        logging.info('Adversarial batch is in zip archive %s',
                     adv_images_files[0])
        shell_call([
            'unzip', os.path.join(input_dir, adv_images_files[0]),
            '-d', input_dir
        ])
        os.remove(os.path.join(input_dir, adv_images_files[0]))
        adv_images_files = os.listdir(input_dir)
      logging.info('%d adversarial images copied', len(adv_images_files))
    except (subprocess.CalledProcessError, IOError) as e:
      raise WorkerError('Can''t copy adversarial batch locally', e)
    # prepare output directory
    if os.path.exists(LOCAL_OUTPUT_DIR):
      sudo_remove_dirtree(LOCAL_OUTPUT_DIR)
    os.mkdir(LOCAL_OUTPUT_DIR)
    output_filname = os.path.join(LOCAL_OUTPUT_DIR, 'result.csv')
    # run defense
    elapsed_time_sec = defense.run(input_dir, output_filname)
    # evaluate defense result
    batch_result = eval_lib.analyze_one_classification_result(
        storage_client=None,
        file_path=output_filname,
        adv_batch=self.adv_batches.data[adversarial_batch_id],
        dataset_batches=self.dataset_batches,
        dataset_meta=self.dataset_meta)
    # copy result of the defense into storage
    try:
      shell_call([
          'gsutil', 'cp', output_filname,
          os.path.join('gs://', self.storage_bucket, cloud_result_path)
      ])
    except subprocess.CalledProcessError as e:
      raise WorkerError('Can''t result to Cloud Storage', e)
    return elapsed_time_sec, submission_id, batch_result

  def run_defenses(self):
    """Method which evaluates all defense work.

    In a loop this method queries not completed defense work,
    picks one defense work and runs it.
    """
    logging.info('******** Start evaluation of defenses ********')
    prev_submission_id = None
    need_reload_work = True
    while True:
      # wait until work is available
      if need_reload_work:
        if self.num_defense_shards:
          shard_with_work = self.defense_work.read_undone_from_datastore(
              shard_id=(self.worker_id % self.num_defense_shards),
              num_shards=self.num_defense_shards)
        else:
          shard_with_work = self.defense_work.read_undone_from_datastore()
        logging.info('Loaded %d records of undone work from shard %s',
                     len(self.defense_work), str(shard_with_work))
      if not self.defense_work.work:
        logging.info('Work is not populated, waiting...')
        time.sleep(SLEEP_TIME)
        continue
      if self.defense_work.is_all_work_competed():
        logging.info('All defense work completed.')
        break
      # download all defense data and dataset
      self.fetch_defense_data()
      need_reload_work = False
      # pick piece of work
      work_id = self.defense_work.try_pick_piece_of_work(
          self.worker_id, submission_id=prev_submission_id)
      if not work_id:
        need_reload_work = True
        logging.info('Failed to pick work, waiting...')
        time.sleep(SLEEP_TIME_SHORT)
        continue
      logging.info('Selected work_id: %s', work_id)
      # execute work
      try:
        elapsed_time_sec, prev_submission_id, batch_result = (
            self.run_defense_work(work_id))
        logging.info('Work %s is done', work_id)
        # indicate that work is completed
        is_work_update = self.defense_work.update_work_as_completed(
            self.worker_id, work_id,
            other_values={'elapsed_time': elapsed_time_sec,
                          'stat_correct': batch_result[0],
                          'stat_error': batch_result[1],
                          'stat_target_class': batch_result[2],
                          'stat_num_images': batch_result[3]})
      except WorkerError as e:
        logging.info('Failed to run work:\n%s', str(e))
        if str(e).startswith('Docker returned non-zero retval'):
          logging.info('Running nvidia-docker to ensure that GPU works')
          shell_call(['nvidia-docker', 'run', '--rm', 'nvidia/cuda',
                      'nvidia-smi'])
        is_work_update = self.defense_work.update_work_as_completed(
            self.worker_id, work_id, error=str(e))
      if not is_work_update:
        logging.warning('Can''t update work "%s" as completed by worker %d',
                        work_id, self.worker_id)
        need_reload_work = True
    logging.info('******** Finished evaluation of defenses ********')

  def run_work(self):
    """Run attacks and defenses"""
    if os.path.exists(LOCAL_EVAL_ROOT_DIR):
      sudo_remove_dirtree(LOCAL_EVAL_ROOT_DIR)
    self.run_attacks()
    self.run_defenses()


def main(args):
  """Main function which runs worker."""
  title = '## Starting evaluation of round {0} ##'.format(args.round_name)
  logging.info('\n'
               + '#' * len(title) + '\n'
               + '#' * len(title) + '\n'
               + '##' + ' ' * (len(title)-2) + '##' + '\n'
               + title + '\n'
               + '#' * len(title) + '\n'
               + '#' * len(title) + '\n'
               + '##' + ' ' * (len(title)-2) + '##' + '\n')
  if args.blacklisted_submissions:
    logging.warning('BLACKLISTED SUBMISSIONS: %s',
                    args.blacklisted_submissions)
  random.seed()
  logging.info('Running nvidia-docker to ensure that GPU works')
  shell_call(['docker', 'run', '--runtime=nvidia',
              '--rm', 'nvidia/cuda', 'nvidia-smi'])
  eval_worker = EvaluationWorker(
      worker_id=args.worker_id,
      storage_client=eval_lib.CompetitionStorageClient(
          args.project_id, args.storage_bucket),
      datastore_client=eval_lib.CompetitionDatastoreClient(
          args.project_id, args.round_name),
      storage_bucket=args.storage_bucket,
      round_name=args.round_name,
      dataset_name=args.dataset_name,
      blacklisted_submissions=args.blacklisted_submissions,
      num_defense_shards=args.num_defense_shards)
  eval_worker.run_work()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Worker which executes work.')
  parser.add_argument('--worker_id',
                      required=True,
                      type=int,
                      help='Numerical ID of the worker.')
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
  parser.add_argument('--num_defense_shards',
                      default=10,
                      required=False,
                      help='Number of defense shards')
  parser.add_argument('--log_file',
                      default='',
                      required=False,
                      help='Location of the logfile.')
  worker_args = parser.parse_args()
  logging_args = {
      'format':
      '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s -- %(message)s',
      'level': logging.INFO,
      'datefmt': '%Y-%m-%d %H:%M:%S',
  }
  if worker_args.log_file:
    logging_args['filename'] = worker_args.log_file
    logging_args['filemode'] = 'a'
  logging.basicConfig(**logging_args)
  main(worker_args)
