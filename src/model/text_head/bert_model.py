import tensorflow as tf
from src.model.text_head.bert_base import BertModel, BertConfig


class BERT(object):
    def __init__(self, bert_config, bert_emb_encode_size, reuse_variables=tf.AUTO_REUSE):
        self.reuse_variables = reuse_variables
        self.bert_emb_encode_size = bert_emb_encode_size
        self.bert_config = BertConfig(**bert_config)

    def __call__(self, input_ids, is_training):
        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        print(input_mask.shape)
        bert_model = BertModel(config=self.bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               reuse_variables=self.reuse_variables)
        text_features = bert_model.get_pooled_output()
        # sequence_output的维度是[batch_size, seq_len, embed_dim]
        # text_features = bert_model.get_sequence_output()
        # avg_pooled = tf.reduce_mean(text_features, 1)
        # max_poolped = tf.reduce_max(text_features, 1)
        # text_features = tf.concat([avg_pooled, max_poolped], 1)
        # tf.print(text_features.shape)
        text_features = tf.layers.dense(text_features, self.bert_emb_encode_size, activation=None, name='text_features',
                                        reuse=self.reuse_variables)
        text_features = tf.layers.batch_normalization(text_features, training=is_training, reuse=self.reuse_variables)
        return text_features


def load_pretrained_model():
    text_pretrained_model = "/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/MultiModal-Tagging/pretrained/albert/albert_model.ckpt"
    assignment_map, _ = train_util.get_assignment_map_from_checkpoint(tf.global_variables(),
                                                                      text_pretrained_model,
                                                                      var_prefix='tower/text/',
                                                                      show=True)
    tf.train.init_from_checkpoint(text_pretrained_model, assignment_map)
    print("load text_pretrained_model: {}".format(text_pretrained_model))


if __name__ == '__main__':
    import yaml
    from utils import train_util
    from src.dataloader.preprocess.text_preprocess import Preprocess

    text = "场景"
    process = Preprocess(
        vocab="/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/MultiModal-Tagging/pretrained/albert/vocab.txt",
        max_len=128,
        co_occurrence_file=None, label_feature_file=None,
    )
    input_ids = process(text)
    input_ids = tf.cast(input_ids, tf.int32)
    input_ids = tf.reshape(input_ids, shape=(-1, 1))
    print(input_ids)

    bert_config = {
         "attention_probs_dropout_prob": 0.1,
         "hidden_act": "gelu",
         "hidden_dropout_prob": 0.1,
         "hidden_size": 768,
         "initializer_range": 0.02,
         "intermediate_size": 3072,
         "max_position_embeddings": 512,
         "num_attention_heads": 12,
         "num_hidden_layers": 12,
         "type_vocab_size": 2,
         "vocab_size": 21128
    }
    model = BERT(bert_config=bert_config, bert_emb_encode_size=1024)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # sess.run(init)
        # print(sess.run(tf.global_variables_initializer()))
        # sess.run(tf.local_variables_initializer())
        sess.run(load_pretrained_model())
        print(sess.run(model(input_ids, is_training=True)))

    # g = tf.Graph()
    # with g.as_default():
    #     model = BERT(**model_config['text_head_params'])
    #
    #     with tf.Session(graph=g) as sess:
    #         tf.global_variables_initializer().run()
    #         print(sess.run(model(input_ids, is_training=False)))



