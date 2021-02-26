#!/bin/bash
#
# Copies baselines to directory which correspond to current round
#

# Import config.sh
source "$( dirname "${BASH_SOURCE[0]}" )/config.sh"
check_dataset_and_round_set

# Copy baselines
gsutil cp gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/baselines/defense/* \
    gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/${ROUND_NAME}/submissions/defense/
gsutil cp gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/baselines/targeted/* \
    gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/${ROUND_NAME}/submissions/targeted/
gsutil cp gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/baselines/nontargeted/* \
    gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/${ROUND_NAME}/submissions/nontargeted/
