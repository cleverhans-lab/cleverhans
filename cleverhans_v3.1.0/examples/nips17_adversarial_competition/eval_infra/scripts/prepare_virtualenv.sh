#!/bin/bash
#
# Script prepares python virtualenv with all necessary libraries
#

# Fail on first error
set -e

# cd to script subdirectory
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Read variables from config.sh
source config.sh

################################################################################
# Create virtual env
################################################################################

cd ~/
rm -rf ~/.virtualenv/${VIRTUALENV_NAME}
virtualenv --system-site-packages ~/.virtualenv/${VIRTUALENV_NAME}

################################################################################
# Install packages into virtualenv
################################################################################

source ~/.virtualenv/${VIRTUALENV_NAME}/bin/activate
pip install --upgrade google-api-python-client
pip install google-cloud
pip install pandas
pip install Pillow
pip install urllib3[secure]
deactivate

echo "ALL DONE!"
