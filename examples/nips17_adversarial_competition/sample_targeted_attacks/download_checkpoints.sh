#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download inception v3 checkpoint for step_target_class attack.
cd "${SCRIPT_DIR}/step_target_class/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz

# Another copy of inception v3 checkpoint for iter_target_class attack
mv inception_v3_2016_08_28.tar.gz "${SCRIPT_DIR}/iter_target_class/"
cd "${SCRIPT_DIR}/iter_target_class/"
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz

