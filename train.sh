#!/usr/bin/env bash

curdir=`pwd`
echo ${curdir}
#
video_dir=${curdir}/dataset/videos
audio_feat_dir=${curdir}/dataset/audio_feat
text_feat_dir=${curdir}/dataset/text_feat
video_feat_dir=${curdir}/dataset/video_feat
img_feat_dir=${curdir}/dataset/img_feat
ground_truth_file=${curdir}/dataset/tagging_info.txt

echo "#####################################"
echo "
    video_dir=${video_dir}
    video_feat_dir=${video_feat_dir}
    audio_feat_dir=${audio_feat_dir}
    text_feat_dir=${text_feat_dir}
    img_feat_dir=${img_feat_dir}
    gt=${ground_truth_file}
"
echo "#####################################"


python generate_datafile.py --video_dir ${video_dir} \
                            --video_feat_dir ${video_feat_dir} \
                            --audio_feat_dir ${audio_feat_dir} \
                            --text_feat_dir ${text_feat_dir} \
                            --img_feat_dir ${img_feat_dir} \
                            --gt ${ground_truth_file}

sudo chmod a+x ./run.sh && ./run.sh train config/train.yaml

