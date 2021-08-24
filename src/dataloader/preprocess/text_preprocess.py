from src.dataloader.preprocess import tokenization
import numpy as np
from src.dataloader.preprocess.label_feature_preprocess import Preprocess as LabelFeaturePreprocess
from src.dataloader.preprocess.co_occurrence_preprocess import Preprocess as CoOccPreprocess
import os
import json


class Preprocess:

    def __init__(self, vocab, max_len, co_occurrence_file, label_feature_file, is_training=False):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab)
        self.max_len = max_len
        self.is_training = is_training
        self.co_occ_preprocess = CoOccPreprocess(co_occurrence_file)
        self.label_fea_preprocess = LabelFeaturePreprocess(label_feature_file)

    def __call__(self, text_path):
        assert os.path.exists(text_path)

        with open(text_path, 'r') as f:
            text = json.load(f)

        text = text['video_ocr']

        tokens = ['[CLS]'] + self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:self.max_len]
        ids = ids + [0]*(self.max_len-len(ids))
        co_matrix = self.co_occ_preprocess()
        return np.array(ids).astype('int64'), self.label_fea_preprocess(), co_matrix[0], co_matrix[1]


def run_test_process():
    pro = Preprocess(
        vocab='/Users/zhiqianhe/MyProjects/腾讯广告算法/new_txad/MultiModal-Tagging/pretrained/albert/vocab.txt',
        max_len=256,
        co_occurrence_file='/Users/zhiqianhe/MyProjects/腾讯广告算法/new_txad/MultiModal-Tagging/dataset/co_occurrence.npy',
        label_feature_file='/Users/zhiqianhe/MyProjects/腾讯广告算法/new_txad/MultiModal-Tagging/dataset/label_feature.npy'
    )
    text_path = '/Users/zhiqianhe/MyProjects/腾讯广告算法/new_txad/MultiModal-Tagging/dataset/tagging/tagging_dataset_train_5k/text_txt/tagging/0af883e5052f18eeaf03a5b281b35497.txt'

    output = pro(text_path)
    print(output)


if __name__ == '__main__':
    run_test_process()
