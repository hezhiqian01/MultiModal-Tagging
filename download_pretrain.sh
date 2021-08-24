#!/usr/bin/env bash

JUPYTER_ROOT=/home/tione/notebook
if [ ! -d "${JUPYTER_ROOT}" ]; then
  echo "JUPYTER_ROOT= ${JUPYTER_ROOT}, not exists, exit"
  exit 1
fi
# CODE ROOT
CODE_ROOT=${JUPYTER_ROOT}/MultiModal-Tagging

mkdir ${CODE_ROOT}/pretrained_models
cd ${CODE_ROOT}/pretrained_models

wget https://taac1-1304126907.cos.ap-guangzhou.myqcloud.com/pretrained_models.tar.gz
tar -xvf pretrained_models.tar.gz