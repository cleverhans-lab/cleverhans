#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download inception v3 checkpoint into base_inception_model subdirectory
cd "${SCRIPT_DIR}/base_inception_model/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz

# Download adversarially trained inception v3 checkpoint
# into adv_inception_v3 subdirectory
cd "${SCRIPT_DIR}/adv_inception_v3/"
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz

# Download ensemble adversarially trained inception resnet v2 checkpoint
# into ens_adv_inception_resnet_v2 subdirectory
cd "${SCRIPT_DIR}/ens_adv_inception_resnet_v2/"
wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
