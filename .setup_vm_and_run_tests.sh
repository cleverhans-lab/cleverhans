#!/bin/sh
set -e
# Run update once so apt-get will work at all
sudo apt-get update
# Install apt-add-repository
sudo apt-get install -y software-properties-common
# Add universe repository so python-pip is available
sudo apt-add-repository universe
# Run update again now that universe is a source
apt-get update
apt-get -y install curl
apt-get install -y wget
rm -rf /var/lib/apt/lists/*

# code below is taken from http://conda.pydata.org/docs/travis.html
# We do this conditionally because it saves us some downloading if the
# version is the same.
export CLOUD_BUILD_PYTHON_VERSION=2.7
if [[ "$CLOUD_BUILD_PYTHON_VERSION" == "2.7" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
else
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
export TENSORFLOW_V="1.8.0"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a

conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy pyqt=4.11 matplotlib pandas h5py six mkl-service
# Enable `conda activate`
sudo ln -s /root/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
conda activate test-environment

# install TensorFlow
if [[ "$CLOUD_BUILD_PYTHON_VERSION" == "2.7" && "$TENSORFLOW_V" == "1.4.1" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp27-none-linux_x86_64.whl;
elif [[ "$COULD_BUILD_PYTHON_VERSION" == "2.7" && "$TENSORFLOW_V" == "1.8.0" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl;
elif [[ "$CLOUD_BUILD_PYTHON_VERSION" == "3.5" && "$TENSORFLOW_V" == "1.4.1" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl;
elif [[ "$CLOUD_BUILD_PYTHON_VERSION" == "3.5" && "$TENSORFLOW_V" == "1.8.0" ]]; then
  pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl;
fi

time pip install -q -e ".[test]"
PYTORCH=True
if [[ "$PYTORCH" == True ]]; then
  pip install torch==0.4.0 torchvision==0.2.1 -q;
fi
# workaround for version incompatibility between the scipy version in conda
# and the system-provided /usr/lib/x86_64-linux-gnu/libstdc++.so.6
# by installing a conda-provided libgcc and adding it to the load path
conda install libgcc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/travis/miniconda/envs/test-environment/lib

# install serialization dependencies
pip install joblib
# install dependencies for adversarial competition eval infra tests
pip install google-cloud==0.33.1
pip install Pillow
# Style checks
pip install pylint
