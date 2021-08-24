#!/usr/bin/env bash


if [ -z "$1" ]; then
    echo "model_path must be provided!"
    exit 1
else
    model_path=$1
fi


if [ -z "$2" ]; then
    echo "output_path is empty, use default="
    output_path=results/inference_result.json
else
    output_path=$2
fi

video_dir="test_dataset/videos"
feat_dir="test_dataset/"

curdir=`pwd`
echo ${curdir}

echo "############################################################"
echo "video_dir=${video_dir}"
echo "model_path=${model_path}"
echo "output_path=${output_path}"
echo "feat_dir=${feat_dir}"
echo "############################################################"

sudo chmod a+x ./run.sh && ./run.sh test ${model_path} ${output_path} ${video_dir} ${feat_dir}

