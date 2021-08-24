#!/usr/bin/env bash

# init jax env
#chmod a+x ./init_jax.sh && ./init.sh

# init tagging env
#chmod a+x ./init_tagging.sh && ./init_tagging.sh


# #################### get env directories
# CONDA_ROOT
CONDA_CONFIG_ROOT_PREFIX=$(conda config --show root_prefix)
echo "CONDA_CONFIG_ROOT_PREFIX= ${CONDA_CONFIG_ROOT_PREFIX}"
get_conda_root_prefix() {
  TMP_POS=$(awk -v a="${CONDA_CONFIG_ROOT_PREFIX}" -v b="/" 'BEGIN{print index(a, b)}')
  TMP_POS=$((TMP_POS-1))
  if [ $TMP_POS -ge 0 ]; then
    echo "${CONDA_CONFIG_ROOT_PREFIX:${TMP_POS}}"
  else
    echo ""
  fi
}
CONDA_ROOT=$(get_conda_root_prefix)
if [ ! -d "${CONDA_ROOT}" ]; then
  echo "CONDA_ROOT= ${CONDA_ROOT}, not exists, exit"
  exit 1
fi

# CONDA ENV
CONDA_NEW_ENV=taac2021-tagging
# JUPYTER_ROOT
JUPYTER_ROOT=/home/tione/notebook
if [ ! -d "${JUPYTER_ROOT}" ]; then
  echo "JUPYTER_ROOT= ${JUPYTER_ROOT}, not exists, exit"
  exit 1
fi
# CODE ROOT
CODE_ROOT=${JUPYTER_ROOT}/MultiModal-Tagging
if [ ! -d "${CODE_ROOT}" ]; then
  echo "CODE_ROOT= ${CODE_ROOT}, not exists, exit"
  exit 1
fi
# OS RELEASE
OS_ID=$(awk -F= '$1=="ID" { print $2 ;}' /etc/os-release)
OS_ID=${OS_ID//"\""/""}

echo "CONDA_ROOT= ${CONDA_ROOT}"
echo "CONDA_NEW_ENV= ${CONDA_NEW_ENV}"
echo "JUPYTER_ROOT= ${JUPYTER_ROOT}"
echo "CODE_ROOT= ${CODE_ROOT}"
echo "OS_ID= ${OS_ID}"

# #################### obviously set $1 to be 'run' to run ./init.sh
if [ -z "$1" ]; then
  ACTION="check"
else
  ACTION=$(echo "$1" | tr '[:upper:]' '[:lower:]')
fi
if [ "${ACTION}" != "run" ]; then
  echo "[Info] you don't set the ACTION as 'run', so just check the environment"
  exit 0
fi

# #################### install cuda
currDir=$PWD
cd ~
wget https://taac-1304126907.cos.ap-nanjing.myqcloud.com/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
export LD_LIBRARY_PATH=/usr/local/cuda-10.1
echo "install cuda 10.1 done."
cd ${currDir}

# #################### install system libraries
if [ "${OS_ID}" == "ubuntu" ]; then
  echo "[Info] installing system libraries in ${OS_ID}"
  sudo apt-get update
  sudo apt-get install -y apt-utils
  sudo apt-get install -y libsndfile1-dev ffmpeg
elif [ "${OS_ID}" == "centos" ]; then
  echo "[Info] installing system libraries in ${OS_ID}"
  yum install -y libsndfile libsndfile-devel ffmpeg ffmpeg-devel
else
  echo "[Warning] os not supported for ${OS_ID}"
  exit 1
fi

conda config --show channels
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda create --prefix ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV} -y cudatoolkit=10.0 cudnn=7.6.0 python=3.7 ipykernel
conda activate ${JUPYTER_ROOT}/envs/${CONDA_NEW_ENV}
conda info --envs

# #################### create jupyter kernel
# create a kernel for conda env
python -m ipykernel install --user --name ${CONDA_NEW_ENV} --display-name "TAAC2021 (${CONDA_NEW_ENV})"

# #################### install python libraries
# install related libraries
#cd ${CODE_ROOT}/MultiModal-Tagging || exit
pwd

# install jax
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.67+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# install other libs
pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
pip install -r requirement.txt
pip install tensorflow-gpu==1.14 opencv-python torch==1.2.0 sklearn torch imgaug


# check tensorflow GPU
python -c "import tensorflow as tf; tf.test.gpu_device_name()"
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
# check library versions
echo "[TensorFlow]"
python -c "import tensorflow as tf; print(tf.__version__)"
echo "[NumPy]"
python -c "import numpy as np; print(np.__version__)"
echo "[Torch]"
python -c "import torch; print(torch.__version__)"
echo "[OpenCV]"
python -c "import cv2; print(cv2.__version__)"
echo "[Jax Jaxlib]"
python -c "import jax; print(jax.local_devices())"

