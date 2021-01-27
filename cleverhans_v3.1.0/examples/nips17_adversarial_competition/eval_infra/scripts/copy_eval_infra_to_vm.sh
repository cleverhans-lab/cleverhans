#!/bin/bash
#
# Helper script which copies evaluation infrastructure to VM
# Usage:
#   copy_eval_infra_to_vm.sh VM_NAME
#

# fail on first error
set -e

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

# Read variables from config.sh
source "${SCRIPT_DIR}/config.sh"

# Check arguments
VM_NAME=$1
if [[ -z ${VM_NAME} ]]; then
  echo "Invalid usage, please provide name of VM:"
  echo "  copy_eval_infra_to_vm.sh VM_NAME"
  exit 1
fi

# Archive evaluation infrastucture
TMP_DIR=$(mktemp -d)
cd ${SCRIPT_DIR}/..
zip -r ${TMP_DIR}/eval_infra.zip ./

# Copy archive to VM
scp_cloud_vm ${TMP_DIR}/eval_infra.zip "${GOOGLE_CLOUD_VM_USERNAME}@${VM_NAME}:~/"

# Unpack eval_infra
run_ssh_command ${VM_NAME} "rm -rf eval_infra"
run_ssh_command ${VM_NAME} "unzip eval_infra.zip -d eval_infra"

# Cleanup
run_ssh_command ${VM_NAME} "rm eval_infra.zip"
rm -r ${TMP_DIR}

