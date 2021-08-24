import os
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
from PIL import Image
from src.feats_extract.imgfeat_extractor.vit_jax import models, checkpoint
from src.feats_extract.imgfeat_extractor.vit_jax.configs import models as models_config


class VideoTransformerExtractor(object):

    def __init__(self):
        model_name = 'ViT-B_16'
        # pretrained_model_file = os.path.join('pretrained', 'vit/ViT-B_16-224.npz')
        pretrained_model_file = "pretrained_models/vit/imagenet21k+imagenet2012_ViT-B_16.npz"

        model_config = models_config.MODEL_CONFIGS[model_name]
        print(model_config)
        self.model = models.VisionTransformer(num_classes=1000, **model_config)
        self.params = checkpoint.load(pretrained_model_file)

    def _resize(self, frame_rgb):
        img = Image.fromarray(frame_rgb).resize(size=(384, 384))
        arr = np.expand_dims(np.array(img), axis=0)
        return arr

    def extract_rgb_frame_features(self, frame_rgb):
        arr = self._resize(frame_rgb)
        frame_features, = self.model.apply(dict(params=self.params), (arr / 128 - 1)[None, ...], train=False)
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
            frame_features_batch = self.model.apply(dict(params=self.params), (frame_rgb_batch / 128 - 1), train=False)
            for _jdx in range(frame_features_batch.shape[0]):
                frame_features_list.append(frame_features_batch[_jdx, :])
        return frame_features_list

