import os
import argparse
import random


def write2file(data, filename):
    with open(filename, 'w') as f:
        for one in data:
            for line in one:
                f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--video_feat_dir', type=str, required=True)
    parser.add_argument('--audio_feat_dir', type=str, required=True)
    parser.add_argument('--text_feat_dir', type=str, required=True)
    parser.add_argument('--img_feat_dir', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)

    args = parser.parse_args()

    video_dir = args.video_dir
    video_feat_dir = args.video_feat_dir
    audio_feat_dir = args.audio_feat_dir
    text_feat_dir = args.text_feat_dir
    img_feat_dir = args.img_feat_dir
    gt = args.gt

    gt_dict = {}
    with open(gt, 'r') as f:
        for line in f:
            line = line.strip()
            vid, tags = line.split('\t')
            gt_dict[vid.split('.')[0]] = tags

    train_data = []
    val_data = []
    total_data = []
    for video in os.listdir(video_dir):
        if video.endswith('mp4'):
            one_data = []
            vid = video.split('.')[0]
            video_feat_path = os.path.join(video_feat_dir, '{}.npy'.format(vid))
            audio_feat_path = os.path.join(audio_feat_dir, '{}.npy'.format(vid))
            img_feat_path = os.path.join(img_feat_dir, '{}.jpg'.format(vid))
            text_feat_path = os.path.join(text_feat_dir, '{}.txt'.format(vid))

            one_data.append(video_feat_path + '\n')
            one_data.append(audio_feat_path + '\n')
            one_data.append(img_feat_path + '\n')
            one_data.append(text_feat_path + '\n')
            one_data.append(gt_dict[vid] + '\n')
            one_data.append('\n')
            total_data.append(one_data)
    random.shuffle(total_data)

    for i, one in enumerate(total_data):
        if i % 10 == 0:
            val_data.append(one)
        else:
            train_data.append(one)

    write2file(train_data, 'dataset/train.txt')
    write2file(val_data, 'dataset/val.txt')


if __name__ == '__main__':
    main()

