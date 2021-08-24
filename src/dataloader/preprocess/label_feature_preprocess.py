import numpy as np


class Preprocess(object):

    def __init__(self, label_feature_file, is_training=False):

        self.label_feature = np.load(label_feature_file)
        self.is_training = is_training

    def __call__(self, *args, **kwargs):
        return self.label_feature
