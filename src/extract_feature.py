from src.feats_extract.multimodal_feature_extract import MultiModalFeatureExtract
import os
import datetime
import argparse
import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgfeat_extractor', type=str, default='vit', help="vit, Youtube8M, InceptionResnetV2")
    parser.add_argument('--extract_video', type=bool, default=True, help='extract video feat or not')
    # parser.add_argument('--extract_audio', type=bool, default=True, help='extract video feat or not')
    # parser.add_argument('--extract_text', type=bool, default=True, help='extract video feat or not')
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--video_out_dir', type=str, default="dataset/vit_features")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    video_dir = args.video_dir
    video_out_dir = args.video_out_dir
    extractor = MultiModalFeatureExtract(batch_size=32,
                                         imgfeat_extractor='vit',
                                         extract_video=True,
                                         extract_audio=False,
                                         extract_text=False,
                                         )
    # video_dir = "/home/tione/notebook/VideoStructuring/dataset/videos/train_5k_A"
    # video_out_dir = "/home/tione/notebook/vit_L_features/"

    if not os.path.isdir(video_out_dir):
        os.makedirs(video_out_dir)

    videos = os.listdir(video_dir)
    for video in tqdm.tqdm(videos):
        vid = video.split('.')[0]
        video_path = os.path.join(video_dir, video)
        frame_npy_path = os.path.join(video_out_dir, "{}.npy".format(vid))
        if os.path.exists(frame_npy_path):
            continue
        extractor.extract_feat(video_path, frame_npy_path=frame_npy_path)
        # print("extract {} done".format(vid))
    print("finished at {}".format(datetime.datetime.now()))


if __name__ == '__main__':
    main()


