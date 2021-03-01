#!/bin/bash
#
# Script which starts evaluation of the competition on all given workers.
# Before running this script VMs with workers should be on, but worker binary
# should not be running on the VMs. If unsure, reload all VMs (via Google Cloud
# web UI) before running this script.
#
# Usage:
#   start_workers.sh INDICES
# Where INDICES is a string with space separated indices of workers to start
#
# Example:
#   start_workers.sh "1 3 4"  # start workers with indices 1, 3 and 4
#   start_workers.sh "$(seq 1 5)"  # start workers with indices 1, 2, 3, 4, 5
#

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd ${SCRIPT_DIR}
source config.sh
check_cloud_compute_zone_set
check_dataset_and_round_set

# Check argument
INDICES=$1
if [[ -z ${INDICES} ]] || [[ "$#" -ne 1 ]]; then
  echo "Invalid usage, please provide list of indices as single argument:"
  echo "  create_workers.sh \"INDICES\""
  exit 1
fi

# Start workers
for idx in ${INDICES}
do
  MACHINE_NAME=`printf "worker-%03g" ${idx}`
  echo "Starting worker ${MACHINE_NAME}"

  # Copy current code of evaluation infrastructure to the worker
  ./copy_eval_infra_to_vm.sh ${MACHINE_NAME}

  # Start worker
  run_ssh_command ${MACHINE_NAME} \
    "eval_infra/code/start_worker_in_tmux.sh ${idx}"

  # Sleep few seconds, so workers will be started with some delay.
  # This helps to reduce contention on Google Cloud Datastore.
  sleep 5
done
