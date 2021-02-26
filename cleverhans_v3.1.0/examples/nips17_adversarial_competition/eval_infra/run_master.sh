#!/bin/bash
#
# Script runs master.
#
# Usage:
#   run_master.sh COMMAND
#

# Get master command from arguments to run_master.sh
COMMAND=$1
if [ -z ${COMMAND} ]; then
  COMMAND=status
fi

# cd to script subdirectory
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Read variables from config.sh
source scripts/config.sh
check_dataset_and_round_set

# Create results directory if it does not exist
mkdir -p ${MASTER_RESULTS_DIR}

# Execute master in proper virtualenv
cd code
source ~/.virtualenv/${VIRTUALENV_NAME}/bin/activate
# NOTE: if you want to use only 30 images from the dataset with 10 images per
# batch then add --limited_dataset flag to following command
python -B master.py \
  ${COMMAND} \
  --project_id="${GOOGLE_CLOUD_PROJECT_ID}" \
  --storage_bucket="${GOOGLE_CLOUD_STORAGE_BUCKET}" \
  --round_name="${ROUND_NAME}" \
  --dataset_name="${DATASET}" \
  --results_dir="${MASTER_RESULTS_DIR}" \
  --num_defense_shards="${NUM_DEFENSE_SHARDS}" \
  --log_file="${MASTER_RESULTS_DIR}/log.txt" \
  --verbose
deactivate
