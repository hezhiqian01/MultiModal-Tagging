from __future__ import unicode_literals
import sys, os
import numpy as np
import cv2
import time
import tensorflow as tf
import json

from src.feats_extract.imgfeat_extractor.youtube8M_extractor import YouTube8MFeatureExtractor
from src.feats_extract.imgfeat_extractor.finetuned_resnet101 import FinetunedResnet101Extractor
from src.feats_extract.txt_extractor.text_requests import VideoASR, VideoOCR, ImageOCR
from src.feats_extract.audio_extractor import vggish_input, vggish_params, vggish_postprocess, vggish_slim


FRAMES_PER_SECOND = 1
PCA_PARAMS = "pretrained/vggfish/vggish_pca_params.npz"  # 'Path to the VGGish PCA parameters file.'
VGGISH_CHECKPOINT = 'pretrained/vggfish/vggish_model.ckpt'
CAP_PROP_POS_MSEC = 0


class MultiModalFeatureExtract(object):
    """docstring for ClassName"""

    def __init__(self, batch_size=1,
                 imgfeat_extractor='Youtube8M',
                 data_aug=False,
                 extract_video=True,
                 extract_audio=True,
                 extract_text=True):
        super(MultiModalFeatureExtract, self).__init__()
        self.extract_video = extract_video
        self.extract_audio = extract_audio
        self.extract_text = extract_text
        self.data_aug = data_aug

        # Video Extract
        if extract_video:
            self.batch_size = batch_size
            if imgfeat_extractor == 'Youtube8M':
                self.extractor = YouTube8MFeatureExtractor(use_batch=batch_size != 1)
            elif imgfeat_extractor == 'FinetunedResnet101':
                self.extractor = FinetunedResnet101Extractor()
            elif imgfeat_extractor == 'InceptionResnetV2':
                from src.feats_extract.imgfeat_extractor.inception_resnet_v2 import InceptionResnetV2Extractor
                self.extractor = InceptionResnetV2Extractor()
            elif imgfeat_extractor == 'vit':
                from src.feats_extract.imgfeat_extractor.video_transformer import VideoTransformerExtractor
                self.extractor = VideoTransformerExtractor()
            else:
                raise NotImplementedError(imgfeat_extractor)

        if extract_audio:
            self.pproc = vggish_postprocess.Postprocessor(PCA_PARAMS)  # audio pca
            self.audio_graph = tf.Graph()
            config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True)
            config.gpu_options.allow_growth = True
            with self.audio_graph.as_default():
                # 音频
                self.audio_sess = tf.Session(graph=self.audio_graph, config=config)
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(self.audio_sess, VGGISH_CHECKPOINT)
            self.features_tensor = self.audio_sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.audio_sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

        if extract_text:
            self.video_ocr_extractor = VideoOCR()
            self.video_asr_extractor = VideoASR()
            self.image_ocr_extractor = ImageOCR()

    def frame_iterator(self, filename, every_ms=1000, max_num_frames=300):
        """Uses OpenCV to iterate over all frames of filename at a given frequency.

        Args:
          filename: Path to video file (e.g. mp4)
          every_ms: The duration (in milliseconds) to skip between frames.
          max_num_frames: Maximum number of frames to process, taken from the
            beginning of the video.

        Yields:
          RGB frame with shape (image height, image width, channels)
        """
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
            print(sys.stderr, 'Error: Cannot open video file ' + filename)
            return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        while num_retrieved < max_num_frames:
            # Skip frames
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            yield frame
            num_retrieved += 1

    def frame_iterator_list(self, filename, every_ms=1000, max_num_frames=300):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
            print(sys.stderr, 'Error: Cannot open video file ' + filename)
            return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        frame_all = []
        while num_retrieved < max_num_frames:
            # Skip frames
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return frame_all

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            frame_all.append(frame[:, :, ::-1])
            num_retrieved += 1

        return frame_all

    def extract_feat(self, test_file,
                     frame_npy_path=None, audio_npy_path=None, txt_file_path=None,
                     image_jpg_path=None, save=True):
        filetype = test_file.split('.')[-1]
        if filetype in ['mp4', 'avi']:
            feat_dict = self.extract_video_feat(test_file, frame_npy_path, audio_npy_path, txt_file_path,
                                                image_jpg_path, save)
        elif filetype in ['jpg', 'png']:
            feat_dict = self.extract_image_feat(test_file)
        else:
            raise NotImplementedError
        if save:
            if 'video' in feat_dict:
                np.save(frame_npy_path, feat_dict['video'])
                print('保存视频特征为{}'.format(frame_npy_path))
            if 'audio' in feat_dict:
                np.save(audio_npy_path, feat_dict['audio'])
                print('保存音频特征为{}'.format(audio_npy_path))
            if 'text' in feat_dict:
                with open(txt_file_path, 'w') as f:
                    f.write(feat_dict['text'])
                print('保存文本特征为{}'.format(txt_file_path))
            if 'image' in feat_dict and filetype == 'mp4':
                cv2.imwrite(image_jpg_path, feat_dict['image'][:, :, ::-1])
        return feat_dict

    def extract_image_feat(self, test_file):
        feat_dict = {}
        feat_dict['image'] = cv2.imread(test_file, 1)[:, :, ::-1]  # convert to rgb

        if self.extract_text:
            start_time = time.time()
            image_ocr = self.image_ocr_extractor.request(test_file)
            feat_dict['text'] = json.dumps({'image_ocr': image_ocr}, ensure_ascii=False)
            end_time = time.time()
            print("text extract cost {} sec".format(end_time - start_time))
        return feat_dict

    def extract_video_feat(self, test_file,
                           frame_npy_path=None, audio_npy_path=None, txt_file_path=None,
                           image_jpg_path=None, save=True):
        feat_dict = {}
        # =============================================视频
        if (frame_npy_path is None or os.path.exists(frame_npy_path)) and save == True:
            pass
        else:
            start_time = time.time()
            if self.batch_size == 1:
                features_arr = []
                for rgb in self.frame_iterator(test_file, every_ms=1000.0 / FRAMES_PER_SECOND):
                    rgb = self.data_argument(rgb)
                    features = self.extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
                    features_arr.append(features)
                feat_dict['video'] = features_arr
            else:
                rgb_list = self.frame_iterator_list(test_file, every_ms=1000.0 / FRAMES_PER_SECOND)
                feat_dict['video'] = self.extractor.extract_rgb_frame_features_list(rgb_list, self.batch_size)
            end_time = time.time()
            print("video extract cost {} sec".format(end_time - start_time))
            # =============================================图片抽帧
        if (image_jpg_path is None or os.path.exists(image_jpg_path)) and save == True:
            pass
        else:
            start_time = time.time()
            rgb_list = self.frame_iterator_list(test_file, every_ms=1000.0 / FRAMES_PER_SECOND)
            feat_dict['image'] = rgb_list[len(rgb_list) // 2]
            end_time = time.time()
            print("image extract cost {} sec".format(end_time - start_time))
        # =============================================音频
        if (audio_npy_path is None or os.path.exists(audio_npy_path)) and save == True:
            # postprocessed_batch = np.load(audio_npy_path)
            pass
        else:
            start_time = time.time()
            output_audio = test_file.replace('.mp4', '.wav')
            if not os.path.exists(output_audio):
                command = 'ffmpeg -loglevel error -i ' + test_file + ' ' + output_audio
                os.system(command)
                # print("audio file not exists: {}".format(output_audio))
                # return
            examples_batch = vggish_input.wavfile_to_examples(output_audio)
            [embedding_batch] = self.audio_sess.run([self.embedding_tensor],
                                                    feed_dict={self.features_tensor: examples_batch})
            feat_dict['audio'] = self.pproc.postprocess(embedding_batch)
            end_time = time.time()
            print("audio extract cost {} sec".format(end_time - start_time))
            # =============================================文本
        if (txt_file_path is None or os.path.exists(txt_file_path)) and save == True:
            pass
        elif self.extract_text:
            start_time = time.time()
            video_ocr = self.video_ocr_extractor.request(test_file)
            video_asr = self.video_asr_extractor.request(test_file)
            feat_dict['text'] = json.dumps({'video_ocr': video_ocr, 'video_asr': video_asr}, ensure_ascii=False)
            print(feat_dict['text'])
            end_time = time.time()
            print("text extract cost {} sec".format(end_time - start_time))
        return feat_dict

    def data_argument(self, rgb_frame):
        if not self.data_aug:
            return rgb_frame
        from imgaug import augmenters as iaa

        seq = iaa.Sequential([
            iaa.CropAndPad(
                px=(0, 30)),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(1),  # horizontally flip 50% of the images
            iaa.Flipud(1),
            iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
            # # iaa.Affine(
            # #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            # #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            # #         rotate=(-45, 45), # rotate by -45 to +45 degrees
            # #         shear=(-16, 16), # shear by -16 to +16 degrees
            # #         order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            # #         cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            # #         mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            # # ),
            iaa.Invert(0.05, per_channel=True),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            iaa.AddToHueAndSaturation((-20, 20)), #############commonly error
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
            iaa.Rot90(1),
            iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
            iaa.Rot90(2),
            iaa.Rot90(3),
        ])
        rgb_frame = seq.augment_image(rgb_frame)
        return rgb_frame


def run_test_extract():
    model = MultiModalFeatureExtract(batch_size=2,
                                     imgfeat_extractor='vit',
                                     extract_video=True,
                                     extract_audio=False,
                                     extract_text=False)
    model.extract_feat(
        test_file="/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/MultiModal-Tagging/src/feats_extract/imgfeat_extractor/90bf818ccdf36b3423f3c9193aa689c4.mp4",
        frame_npy_path='./test.npy'
    )
    print(np.load('./test.npy'))
    # print(res['video'][0].shape)
    # print(len(res['video']))
    # img_file = "/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/MultiModal-Tagging/src/feats_extract/imgfeat_extractor/beautiful_girl.jpg"
    # img = cv2.imread(img_file)

    # img = model.data_argument(img)
    # cv2.imshow('bg', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    run_test_extract()
