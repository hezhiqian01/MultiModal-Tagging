#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "video dir must be provided!"
    exit 1
else
    video_dir=$1
fi

if [ -z "$2" ]; then
    echo "video out dir must be provided!"
    exit 1
else
    video_out_dir=$2
fi


echo "############################################################"
echo "video_dir=${video_dir}"
echo "video_out_dir=${video_out_dir}"
echo "############################################################"

export LD_LIBRARY_PATH="/usr/local/cuda-10.1/"  && ./run.sh extract ${video_dir} ${video_out_dir}

