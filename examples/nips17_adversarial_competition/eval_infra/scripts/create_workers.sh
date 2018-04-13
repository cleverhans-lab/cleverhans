#!/bin/bash
#
# Script which creates virtual machines for all workers.
# Note that all virtual machines are creates in running state, thus you
# will be billed for them. Also this script does not start evaluation,
# you need to use start_workers.sh to actually start evaluation
# of the competition.
#
# Usage:
#   create_workers.sh INDICES
# Where INDICES is a string with space separated indices of workers to create
#
# Example:
#   create_workers.sh "1 3 4"  # create workers with indices 1, 3 and 4
#   create_workers.sh "$(seq 1 5)"  # create workers with indices 1, 2, 3, 4, 5
#

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd ${SCRIPT_DIR}
source config.sh

# Check that variables are set
check_cloud_compute_zone_set
check_cloud_ref_vm_snapshot_set

# Check argument
INDICES=$1
if [[ -z ${INDICES} ]] || [[ "$#" -ne 1 ]]; then
  echo "Invalid usage, please provide list of indices as single argument:"
  echo "  create_workers.sh \"INDICES\""
  exit 1
fi

# Parameters of VM, feel free to change if you need different specs
DISK_SIZE=200  # disk size in Gb
MACHINE_TYPE="n1-highmem-4"  # type of machine, determines CPU and amount of RAM
GPU_TYPE="nvidia-tesla-k80"  # type of GPU

# Create all workers
for idx in ${INDICES}
do
  MACHINE_NAME=`printf "worker-%03g" ${idx}`
  echo "Creating worker ${MACHINE_NAME}"

  gcloud compute --project "${GOOGLE_CLOUD_PROJECT_ID}" \
    disks create "${MACHINE_NAME}" --size "${DISK_SIZE}" \
    --zone "${GOOGLE_CLOUD_COMPUTE_ZONE}" \
    --source-snapshot "${GOOGLE_CLOUD_REF_VM_SNAPSHOT}" --type "pd-standard"

  gcloud beta compute --project "${GOOGLE_CLOUD_PROJECT_ID}" \
    instances create "${MACHINE_NAME}" \
    --zone "${GOOGLE_CLOUD_COMPUTE_ZONE}" \
    --machine-type "${MACHINE_TYPE}" --subnet "default" \
    --maintenance-policy "TERMINATE" \
    --scopes "https://www.googleapis.com/auth/datastore","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/trace.append","https://www.googleapis.com/auth/devstorage.read_write" \
    --accelerator type=${GPU_TYPE},count=1 \
    --min-cpu-platform "Automatic" \
    --disk "name=${MACHINE_NAME},device-name=${MACHINE_NAME},mode=rw,boot=yes,auto-delete=yes"
done
