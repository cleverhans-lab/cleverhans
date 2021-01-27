#!/bin/bash
#
# Constants and variables related to evaluation of the competition.
# Please update them appropriately before running evaluation
#

################################################################################
# Variables
################################################################################

# Google Cloud Project ID
GOOGLE_CLOUD_PROJECT_ID=""

# Google Cloud Storage bucket where all data related to competition are stored
GOOGLE_CLOUD_STORAGE_BUCKET=""

# Name of the zone where VMs are created
GOOGLE_CLOUD_COMPUTE_ZONE=""

# Google Cloud VM username
GOOGLE_CLOUD_VM_USERNAME="${USER}"

# Name of the VM snapshot which is used as reference images for all workers
GOOGLE_CLOUD_REF_VM_SNAPSHOT=""

# Name of the current round
ROUND_NAME=""

# Dataset to use
DATASET=""

# Number of shards to compute defense results
# As a rule of thumb use NUM_DEFENSE_SHARDS equal to NUM_WORKERS/10
# Increase number of shards if you see that workers have periodic
# transient errors while accessing datastore.
NUM_DEFENSE_SHARDS=1

# Local directory where master will save results of the competition
MASTER_RESULTS_DIR="${HOME}/adversarial_competition/results/${ROUND_NAME}"

# Name of the python virtualenv which is used to run the master
VIRTUALENV_NAME="nips_competition_env"

################################################################################
# Check that necessary variables are set
# GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_STORAGE_BUCKET are checked every time
# config is imported.
# Few other variables are checked only when needed by provided helper functions.
################################################################################

if [[ -z ${GOOGLE_CLOUD_PROJECT_ID} ]]; then
  echo "GOOGLE_CLOUD_PROJECT_ID variable must be set in config.sh"
  exit 1
fi

if [[ -z ${GOOGLE_CLOUD_STORAGE_BUCKET} ]]; then
  echo "GOOGLE_CLOUD_STORAGE_BUCKET variable must be set in config.sh"
  exit 1
fi

function check_cloud_compute_zone_set() {
  if [[ -z ${GOOGLE_CLOUD_COMPUTE_ZONE} ]]; then
    echo "GOOGLE_CLOUD_COMPUTE_ZONE variable must be set in config.sh"
    exit 1
  fi
}

function check_cloud_ref_vm_snapshot_set() {
  if [[ -z ${GOOGLE_CLOUD_REF_VM_SNAPSHOT} ]]; then
    echo "GOOGLE_CLOUD_REF_VM_SNAPSHOT variable must be set in config.sh"
    exit 1
  fi
}

function check_dataset_and_round_set() {
  if [[ -z ${DATASET} ]]; then
    echo "DATASET variable must be set in config.sh"
    exit 1
  fi

  if [[ -z ${ROUND_NAME} ]]; then
    echo "DATASET variable must be set in config.sh"
    exit 1
  fi
}

################################################################################
# Helper functions
################################################################################

function run_ssh_command() {
  check_cloud_compute_zone_set

  local machine_name=$1
  local command=$2

  gcloud compute --project "${GOOGLE_CLOUD_PROJECT_ID}" \
    ssh --zone "${GOOGLE_CLOUD_COMPUTE_ZONE}" "${machine_name}" \
    --command="${command}"
}

function scp_cloud_vm() {
  check_cloud_compute_zone_set

  local src=$1
  local dst=$2

  gcloud compute scp \
    --project "${GOOGLE_CLOUD_PROJECT_ID}" \
    --zone "${GOOGLE_CLOUD_COMPUTE_ZONE}" \
    ${src} ${dst}
}
