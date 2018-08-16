#!/bin/bash
#
# Script which copies dataset to VM
# Usage:
#   copy_dataset_to_vm.sh DATASET_LOCAL_DIR DATASET_NAME VM_NAME
# Where:
#   DATASET_LOCAL_DIR - local directory where dataset is located,
#       directory with dataset should contain file "${DATASET_NAME}_dataset.csv"
#       and subdirectory "images" with all dataset images
#   DATASET_NAME - name of the dataset, "dev" or "final"
#   VM_NAME - name of virtual machine where data should be copied into
#

# fail on first error
set -e

# Read variables from config.sh
SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
source "${SCRIPT_DIR}/config.sh"

# Check arguments
DATASET_DIR=$1
DATASET_NAME=$2
VM_NAME=$3

if [[ -z ${DATASET_DIR} ]] || [[ -z ${DATASET_NAME} ]] || [[ -z ${VM_NAME} ]]; then
  echo "Invalid usage, please run:"
  echo "  copy_dataset_to_vm.sh DATASET_LOCAL_DIR DATASET_NAME VM_NAME"
  exit 1
fi

# Archive dataset
TMP_DIR=$(mktemp -d)
cd ${DATASET_DIR}
zip -r ${TMP_DIR}/dataset.zip ./

# Copy archive with dataset to VM
scp_cloud_vm ${TMP_DIR}/dataset.zip "${GOOGLE_CLOUD_VM_USERNAME}@${VM_NAME}:~/"

# Unpack dataset
run_ssh_command ${VM_NAME} "rm -rf competition_data/dataset/${DATASET_NAME}"
run_ssh_command ${VM_NAME} "mkdir -p competition_data/dataset/${DATASET_NAME}"
run_ssh_command ${VM_NAME} "unzip dataset.zip -d competition_data/dataset/${DATASET_NAME}"

# Cleanup
run_ssh_command ${VM_NAME} "rm dataset.zip"
rm -r ${TMP_DIR}

echo "DONE!"
