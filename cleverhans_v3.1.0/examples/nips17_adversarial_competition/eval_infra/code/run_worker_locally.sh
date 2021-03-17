#!/bin/bash
#
# This script starts worker. Generally there is no need to manually runn this
# script, it should be invoked during deployment of the worker to VM.
#
# However if you need to run this script manually, usage is following:
#
#   run_worker_locally.sh WORKER_ID
#
# where WORKER_ID is numerical identifier of the worker. Typically workers are
# identified by numbers from 0 to NUMBER_OF_WORKERS-1
#

# Get worker ID from argumetns to run_worker_locally.sh
WORKER_ID=$1
if [ -z ${WORKER_ID} ]; then
  echo "Worker ID is missing."
  exit 1
fi

# cd to script directory
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Read variables from config.sh
source ../scripts/config.sh
check_dataset_and_round_set

# Run worekr
source ~/.virtualenv/${VIRTUALENV_NAME}/bin/activate
python -B worker.py \
  --worker_id=${WORKER_ID} \
  --project_id="${GOOGLE_CLOUD_PROJECT_ID}" \
  --storage_bucket="${GOOGLE_CLOUD_STORAGE_BUCKET}" \
  --round_name="${ROUND_NAME}" \
  --dataset_name="${DATASET}" \
  --num_defense_shards="${NUM_DEFENSE_SHARDS}" &> ~/log.txt
deactivate
