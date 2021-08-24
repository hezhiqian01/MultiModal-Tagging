import tensorflow as tf
from src.model.text_head.albert_base import BertModel, BertConfig


class ALBERT(object):
    def __init__(self, bert_config_json, bert_emb_encode_size):
        self.bert_emb_encode_size = bert_emb_encode_size
        self.bert_config = BertConfig.from_json_file(bert_config_json)

    def __call__(self, input_ids, is_training):
        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        print(input_mask.shape)
        bert_model = BertModel(config=self.bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               )
        text_features = bert_model.get_pooled_output()

        # sequence_output的维度是[batch_size, seq_len, embed_dim]
        # text_features = bert_model.get_sequence_output()
        # avg_pooled = tf.reduce_mean(text_features, 1)
        # max_poolped = tf.reduce_max(text_features, 1)
        # text_features = tf.concat([avg_pooled, max_poolped], 1)

        # text_features = tf.layers.dense(text_features, self.bert_emb_encode_size, activation=None, name='text_features',
        #                                 reuse=tf.AUTO_REUSE)
        # text_features = tf.layers.batch_normalization(text_features, training=is_training, reuse=tf.AUTO_REUSE)
        return text_features


def load_pretrained_model():
    text_pretrained_model = "/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/MultiModal-Tagging/pretrained/albert_base/albert_model.ckpt"
    assignment_map, _ = train_util.get_assignment_map_from_checkpoint(tf.global_variables(),
                                                                      text_pretrained_model,
                                                                      var_prefix='',
                                                                      show=True)
    tf.train.init_from_checkpoint(text_pretrained_model, assignment_map)
    print("load text_pretrained_model: {}".format(text_pretrained_model))


if __name__ == '__main__':
    import yaml
    from utils import train_util
    from src.dataloader.preprocess.text_preprocess import Preprocess
    import numpy as np

    tags = []
    with open('/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/dataset/label_id.txt', 'r') as f:
        for line in f:
            line = line.strip()
            tags.append(line.split('\t')[0])

    process = Preprocess(
        vocab="/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/MultiModal-Tagging/pretrained/albert/vocab.txt",
        max_len=128,
        co_occurrence_file=None, label_feature_file=None
    )
    inputs = []
    for tag in tags:
        text = "场景"
        input_ids = process(text)
        input_ids = tf.cast(input_ids, tf.int32)
        # input_ids = tf.reshape(input_ids, shape=(-1, 1))
        inputs.append(input_ids)
    inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)
    print(inputs)
    # tf.random.set_random_seed(1)

    model = ALBERT(bert_config_json="/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/MultiModal-Tagging/pretrained/albert_base/albert_config_base.json",
                   bert_emb_encode_size=1024)
    output = model(inputs, is_training=True)
    # ou = tf.get_default_graph().get_tensor_by_name("bert/embeddings/word_embeddings_2:0")
    # print(ou)
    load_pretrained_model()
    saver = tf.train.Saver(max_to_keep=5)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # sess.run()
        # sess.run(init)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        output = sess.run(output)
        print(output.shape)
        np.save("/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/dataset/label_feature.npy", output)
