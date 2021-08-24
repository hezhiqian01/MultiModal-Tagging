# import cv2
import numpy as np
from PIL import Image
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input


class InceptionResnetV2Extractor(object):

    def __init__(self):
        self.model = InceptionResNetV2(include_top=False)

    def _resize(self, frame_rgb):
        img = Image.fromarray(frame_rgb).resize(size=(299, 299))
        arr = np.expand_dims(np.array(img), axis=0)
        return arr

    def _pooling(self, frame_features):
        # frame_features = np.mean(frame_features, axis=(1, 2))
        return frame_features

    def extract_rgb_frame_features(self, frame_rgb):
        arr = self._resize(frame_rgb)

        frame_features = self.model.predict(preprocess_input(arr))
        frame_features = self._pooling(frame_features)
        return frame_features

    def extract_rgb_frame_features_list(self, frame_rgb_list, batch_size):
        input_list = []
        for _idx, frame_rgb in enumerate(frame_rgb_list):
            frame_rgb = self._resize(frame_rgb)
            if _idx % batch_size == 0:
                frame_rgb_batch = frame_rgb
            else:
                frame_rgb_batch = np.concatenate((frame_rgb_batch, frame_rgb), axis=0)
            if (_idx % batch_size == batch_size - 1) or _idx == len(frame_rgb_list) - 1:
                input_list.append(frame_rgb_batch)

        frame_features_list = []
        for frame_rgb_batch in input_list:
            frame_features_batch = self.model.predict(frame_rgb_batch)
            frame_features_batch = self._pooling(frame_features_batch).reshape(-1, 8, 8, 1536)
            for _jdx in range(frame_features_batch.shape[0]):
                frame_features_list.append(frame_features_batch[_jdx, :])
        return frame_features_list

