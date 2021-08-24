import numpy as np


class Preprocess(object):

    def __init__(self, co_occurrence_file, is_training=False):
        self.co_occurrence_matrix = np.load(co_occurrence_file)
        self.is_training = is_training

    def __call__(self, *args, **kwargs):
        return self.co_occurrence_matrix.T, self.co_occurrence_matrix
