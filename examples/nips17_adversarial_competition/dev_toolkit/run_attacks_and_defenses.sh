#!/bin/bash

# exit on first error
set -e

# directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ATTACKS_DIR="${SCRIPT_DIR}/sample_attacks"
TARGETED_ATTACKS_DIR="${SCRIPT_DIR}/sample_targeted_attacks"
DEFENSES_DIR="${SCRIPT_DIR}/sample_defenses"
DATASET_DIR="${SCRIPT_DIR}/dataset/images"
DATASET_METADATA_FILE="${SCRIPT_DIR}/dataset/dev_dataset.csv"
MAX_EPSILON=16

# Prepare working directory and copy all necessary files.
# In particular copy attacks defenses and dataset, so originals won't
# be overwritten.
if [[ "${OSTYPE}" == "darwin"* ]]; then
    WORKING_DIR="/private"$(mktemp -d)
else
    WORKING_DIR=$(mktemp -d)
fi
echo "Preparing working directory: ${WORKING_DIR}"
mkdir "${WORKING_DIR}/attacks"
mkdir "${WORKING_DIR}/targeted_attacks"
mkdir "${WORKING_DIR}/defenses"
mkdir "${WORKING_DIR}/dataset"
mkdir "${WORKING_DIR}/intermediate_results"
mkdir "${WORKING_DIR}/output_dir"
cp -R "${ATTACKS_DIR}"/* "${WORKING_DIR}/attacks"
cp -R "${TARGETED_ATTACKS_DIR}"/* "${WORKING_DIR}/targeted_attacks"
cp -R "${DEFENSES_DIR}"/* "${WORKING_DIR}/defenses"
cp -R "${DATASET_DIR}"/* "${WORKING_DIR}/dataset"
cp "${DATASET_METADATA_FILE}" "${WORKING_DIR}/dataset.csv"

echo "Running attacks and defenses"
python "${SCRIPT_DIR}/run_attacks_and_defenses.py" \
  --attacks_dir="${WORKING_DIR}/attacks" \
  --targeted_attacks_dir="${WORKING_DIR}/targeted_attacks" \
  --defenses_dir="${WORKING_DIR}/defenses" \
  --dataset_dir="${WORKING_DIR}/dataset" \
  --intermediate_results_dir="${WORKING_DIR}/intermediate_results" \
  --dataset_metadata="${WORKING_DIR}/dataset.csv" \
  --output_dir="${WORKING_DIR}/output_dir" \
  --epsilon="${MAX_EPSILON}" \
  --save_all_classification

echo "Output is saved in directory '${WORKING_DIR}/output_dir'"
