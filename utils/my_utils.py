import json
import os
import random
from collections import defaultdict
import numpy as np


# 每个视频作为一条数据
def load_data(filepath):
    res = []
    with open(filepath, "r") as f:
        one_data = []
        for i, line in enumerate(f):
            one_data.append(line)
            if (i+1) % 6 == 0:
                res.append(one_data)
                one_data = []
    return res


def get_vids_from_file(filepath):
    """
    从文件中获取所有的vid
    :param filepath: 训练文件或者验证集文件
    :return:
    """
    vids = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if i % 6 == 0:
                vid = line.strip().split('/')[-1]
                vids.append(vid.replace('npy', 'mp4'))
    return vids


def split_data(data, num):
    """
    将数据集分成10份，每份为500，num代表是取哪一份作为valid data
    """
    train_data = []
    val_data = []

    for i in range(10):
        if i + 1 == num:
            val_data.extend(data[i * 500: i * 500 + 500])
        else:
            train_data.extend(data[i * 500:i * 500 + 500])
    return train_data, val_data


def write2file(data, filepath):
    with open(filepath, 'w') as f:
        for one in data:
            for line in one:
                f.write(line)
    print("write {} Nums. of line to {}".format(len(data), filepath))


def cross_val_data(total_data, outpath):
    """
    10-折交叉验证
    :param total_data:
    :param outpath:
    :return:
    """
    random.shuffle(total_data)
    for i in range(1, 11):
        train_data, val_data = split_data(total_data, i)
        write2file(train_data, os.path.join(outpath, "train_{}.txt".format(i)))
        write2file(val_data, os.path.join(outpath, "val_{}.txt".format(i)))


def generate_dataset_file(old_train_file, old_val_file, video_fea_dir, out_dir):
    """
    利用新提取好的视频特征，生成新的train 和val file
    :param old_train_file:
    :param old_val_file:
    :param video_fea_dir:
    :param out_dir:
    :return:
    """
    vit_train = []
    vit_val = []
    # vit_video_fea_dir = '/home/tione/notebook/vit_L_features'

    train_data = load_data(old_train_file)
    val_data = load_data(old_val_file)

    for i, one_data in enumerate(train_data):
        vid = one_data[0].strip().split('/')[-1]
        one_data[0] = os.path.join(video_fea_dir, vid) + '\n'
        for line in one_data:
            vit_train.append(line)

    print(len(vit_train))

    for i, one_data in enumerate(val_data):
        vid = one_data[0].strip().split('/')[-1]
        one_data[0] = os.path.join(video_fea_dir, vid) + '\n'
        for line in one_data:
            vit_val.append(line)

    vit_train_file = os.path.join(out_dir, 'vit_train.txt')
    vit_val_file = os.path.join(out_dir, 'vit_val.txt')

    with open(vit_train_file, 'w') as f:
        for line in vit_train:
            f.write(line)

    print("write {} Nums. of line to {}".format(len(vit_train), vit_train_file))

    with open(vit_val_file, 'w') as f:
        for line in vit_val:
            f.write(line)
    print("write {} Nums. of line to {}".format(len(vit_val), vit_val_file))


def load_res_file(res_file):
    with open(res_file, 'r') as f:
        content = json.load(f)
    res = {}
    for vid in content:
        labels = content[vid]['result'][0]['labels']
        scores = content[vid]['result'][0]['scores']
        res[vid] = [labels, scores]
    return res


def fusion(*res_file):
    res_num = len(res_file)

    total_res = {}
    for filepath in res_file:
        print(filepath)
        res = load_res_file(filepath)
        for vid in res:
            if vid not in total_res:
                total_res[vid] = defaultdict(list)
            labels, scores = res[vid]
            for i, lab in enumerate(labels):
                total_res[vid][lab].append(float(scores[i]))

    average_res = {}
    # 每个label的取平均
    for vid in total_res:
        average_res[vid] = []
        for lab in total_res[vid]:
            while len(total_res[vid][lab]) < res_num:
                total_res[vid][lab].append(0)
            average_res[vid].append([lab, np.mean(total_res[vid][lab])])

    # 排序取top20
    for vid in average_res:
        average_res[vid].sort(key=lambda x: x[1], reverse=True)
        average_res[vid] = average_res[vid][:20]

    # 重新格式化为原先的输出格式
    final_res = {}
    for vid in average_res:
        final_res[vid] = {
            "result": [{
                'labels': [],
                'scores': []
            }]
        }
        for lab, score in average_res[vid]:
            final_res[vid]['result'][0]['labels'].append(lab)
            final_res[vid]['result'][0]['scores'].append(score)

    return final_res


def split_result(total_result_file, train_file, val_file):
    """
    把所有包含训练集和验证集的预测结果重新分为训练集和测试集
    :param train_result:
    :return:
    """
    train_vids = get_vids_from_file(train_file)
    val_vids = get_vids_from_file(val_file)

    train_res = {}
    val_res = {}

    with open(total_result_file, 'r') as f:
        total = json.load(f)

    for vid in total:
        if vid in train_vids:
            train_res[vid] = total[vid]
        elif vid in val_vids:
            val_res[vid] = total[vid]
        else:
            raise ValueError
    return train_res, val_res


def generate_valid_GT(tagging_info_file, valid_file, valid_tagging_info_file):
    val_vids = get_vids_from_file(valid_file)
    valid_GT = []

    with open(tagging_info_file, 'r') as f:
        for line in f:
            vid = line.strip().split('\t')[0].replace('npy', 'mp4')
            if vid in val_vids:
                valid_GT.append(line)

    with open(valid_tagging_info_file, 'w') as f:
        for line in valid_GT:
            f.write(line)


