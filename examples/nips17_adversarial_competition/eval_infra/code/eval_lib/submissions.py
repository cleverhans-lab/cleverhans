"""Classes and functions to manage submissions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from io import StringIO
import os
from six import iteritems

# Cloud Storage directories
ATTACK_SUBDIR = 'submissions/nontargeted'
TARGETED_ATTACK_SUBDIR = 'submissions/targeted'
DEFENSE_SUBDIR = 'submissions/defense'

# Cloud Datastore entity keys
ATTACKS_ENTITY_KEY = [u'SubmissionType', u'Attacks']
TARGET_ATTACKS_ENTITY_KEY = [u'SubmissionType', u'TargetedAttacks']
DEFENSES_ENTITY_KEY = [u'SubmissionType', u'Defenses']
KIND_SUBMISSION = u'Submission'

# Cloud Datastore ID patterns
ATTACK_ID_PATTERN = u'SUBA{:03}'
TARGETED_ATTACK_ID_PATTERN = u'SUBT{:03}'
DEFENSE_ID_PATTERN = u'SUBD{:03}'

# Constants for __str__
TO_STR_MAX_SUBMISSIONS = 5

ALLOWED_EXTENSIONS = ['.zip', '.tar', '.tar.gz']


def participant_from_submission_path(submission_path):
  """Parses type of participant based on submission filename.

  Args:
    submission_path: path to the submission in Google Cloud Storage

  Returns:
    dict with one element. Element key correspond to type of participant
    (team, baseline), element value is ID of the participant.

  Raises:
    ValueError: is participant can't be determined based on submission path.
  """
  basename = os.path.basename(submission_path)
  file_ext = None
  for e in ALLOWED_EXTENSIONS:
    if basename.endswith(e):
      file_ext = e
      break
  if not file_ext:
    raise ValueError('Invalid submission path: ' + submission_path)
  basename = basename[:-len(file_ext)]
  if basename.isdigit():
    return {'team_id': int(basename)}
  if basename.startswith('baseline_'):
    return {'baseline_id': basename[len('baseline_'):]}
  raise ValueError('Invalid submission path: ' + submission_path)


SubmissionDescriptor = namedtuple('SubmissionDescriptor',
                                  ['path', 'participant_id'])


class CompetitionSubmissions(object):
  """Class which holds information about all submissions.

  All submissions are stored in 3 dictionaries, one for targeted attacks,
  one for non-targeted attacks and one for defenses.
  All submissions are identified using internal competition ID,
  which looks like 'SUB????'. Additionally each submission has external
  identified which could be name of baseline or Kaggle ID.
  External ID only used when list of submissions is formed and when
  scorebored is built. Internal submission IDs are used for all actual
  evaluation. Thus all identifiers are internal IDs unless otherwise noted.
  """

  def __init__(self, datastore_client, storage_client, round_name):
    """Initializes CompetitionSubmissions.

    Args:
      datastore_client: instance of CompetitionDatastoreClient
      storage_client: instance of CompetitionStorageClient
      round_name: name of the round
    """
    self._datastore_client = datastore_client
    self._storage_client = storage_client
    self._round_name = round_name
    # each of the variables is a dictionary,
    # where key - submission ID
    # value - SubmissionDescriptor namedtuple
    self._attacks = None
    self._targeted_attacks = None
    self._defenses = None

  def _load_submissions_from_datastore_dir(self, dir_suffix, id_pattern):
    """Loads list of submissions from the directory.

    Args:
      dir_suffix: suffix of the directory where submissions are stored,
        one of the folowing constants: ATTACK_SUBDIR, TARGETED_ATTACK_SUBDIR
        or DEFENSE_SUBDIR.
      id_pattern: pattern which is used to generate (internal) IDs
        for submissins. One of the following constants: ATTACK_ID_PATTERN,
        TARGETED_ATTACK_ID_PATTERN or DEFENSE_ID_PATTERN.

    Returns:
      dictionary with all found submissions
    """
    submissions = self._storage_client.list_blobs(
        prefix=os.path.join(self._round_name, dir_suffix))
    return {
        id_pattern.format(idx): SubmissionDescriptor(
            path=s, participant_id=participant_from_submission_path(s))
        for idx, s in enumerate(submissions)
    }

  def init_from_storage_write_to_datastore(self):
    """Init list of sumibssions from Storage and saves them to Datastore.

    Should be called only once (typically by master) during evaluation of
    the competition.
    """
    # Load submissions
    self._attacks = self._load_submissions_from_datastore_dir(
        ATTACK_SUBDIR, ATTACK_ID_PATTERN)
    self._targeted_attacks = self._load_submissions_from_datastore_dir(
        TARGETED_ATTACK_SUBDIR, TARGETED_ATTACK_ID_PATTERN)
    self._defenses = self._load_submissions_from_datastore_dir(
        DEFENSE_SUBDIR, DEFENSE_ID_PATTERN)
    self._write_to_datastore()

  def _write_to_datastore(self):
    """Writes all submissions to datastore."""
    # Populate datastore
    roots_and_submissions = zip([ATTACKS_ENTITY_KEY,
                                 TARGET_ATTACKS_ENTITY_KEY,
                                 DEFENSES_ENTITY_KEY],
                                [self._attacks,
                                 self._targeted_attacks,
                                 self._defenses])
    client = self._datastore_client
    with client.no_transact_batch() as batch:
      for root_key, submissions in roots_and_submissions:
        batch.put(client.entity(client.key(*root_key)))
        for k, v in iteritems(submissions):
          entity = client.entity(client.key(
              *(root_key + [KIND_SUBMISSION, k])))
          entity['submission_path'] = v.path
          entity.update(participant_from_submission_path(v.path))
          batch.put(entity)

  def init_from_datastore(self):
    """Init list of submission from Datastore.

    Should be called by each worker during initialization.
    """
    self._attacks = {}
    self._targeted_attacks = {}
    self._defenses = {}
    for entity in self._datastore_client.query_fetch(kind=KIND_SUBMISSION):
      submission_id = entity.key.flat_path[-1]
      submission_path = entity['submission_path']
      participant_id = {k: entity[k]
                        for k in ['team_id', 'baseline_id']
                        if k in entity}
      submission_descr = SubmissionDescriptor(path=submission_path,
                                              participant_id=participant_id)
      if list(entity.key.flat_path[0:2]) == ATTACKS_ENTITY_KEY:
        self._attacks[submission_id] = submission_descr
      elif list(entity.key.flat_path[0:2]) == TARGET_ATTACKS_ENTITY_KEY:
        self._targeted_attacks[submission_id] = submission_descr
      elif list(entity.key.flat_path[0:2]) == DEFENSES_ENTITY_KEY:
        self._defenses[submission_id] = submission_descr

  @property
  def attacks(self):
    """Dictionary with all non-targeted attacks."""
    return self._attacks

  @property
  def targeted_attacks(self):
    """Dictionary with all targeted attacks."""
    return self._targeted_attacks

  @property
  def defenses(self):
    """Dictionary with all defenses."""
    return self._defenses

  def get_all_attack_ids(self):
    """Returns IDs of all attacks (targeted and non-targeted)."""
    return list(self.attacks.keys()) + list(self.targeted_attacks.keys())

  def find_by_id(self, submission_id):
    """Finds submission by ID.

    Args:
      submission_id: ID of the submission

    Returns:
      SubmissionDescriptor with information about submission or None if
      submission is not found.
    """
    return self._attacks.get(
        submission_id,
        self._defenses.get(
            submission_id,
            self._targeted_attacks.get(submission_id, None)))

  def get_external_id(self, submission_id):
    """Returns human readable submission external ID.

    Args:
      submission_id: internal submission ID.

    Returns:
      human readable ID.
    """
    submission = self.find_by_id(submission_id)
    if not submission:
      return None
    if 'team_id' in submission.participant_id:
      return submission.participant_id['team_id']
    elif 'baseline_id' in submission.participant_id:
      return 'baseline_' + submission.participant_id['baseline_id']
    else:
      return ''

  def __str__(self):
    """Returns human readable representation, useful for debugging purposes."""
    buf = StringIO()
    title_values = zip([u'Attacks', u'Targeted Attacks', u'Defenses'],
                       [self._attacks, self._targeted_attacks, self._defenses])
    for idx, (title, values) in enumerate(title_values):
      if idx >= TO_STR_MAX_SUBMISSIONS:
        buf.write('...\n')
        break
      buf.write(title)
      buf.write(u':\n')
      for k, v in iteritems(values):
        buf.write(u'{0} -- {1}   {2}\n'.format(k, v.path,
                                               str(v.participant_id)))
      buf.write(u'\n')
    return buf.getvalue()
