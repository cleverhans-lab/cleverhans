#!/bin/bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Download checkpoints for sample attacks and defenses.
sample_attacks/download_checkpoints.sh
sample_targeted_attacks/download_checkpoints.sh
sample_defenses/download_checkpoints.sh

# Download dataset.
mkdir -p dataset/images
cp ../dataset/dev_dataset.csv dataset/
python ../dataset/download_images.py \
  --input_file=dataset/dev_dataset.csv \
  --output_dir=dataset/images/
