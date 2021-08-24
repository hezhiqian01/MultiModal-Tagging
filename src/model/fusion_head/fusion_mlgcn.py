from src.model.gcn.gcn import GCN
from src.model.fusion_head.fusion_se import SE
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class MLGCN(object):

    def __init__(self,
                 word_dim,
                 intermediary_dim1,
                 output_dim,
                 drop_rate,
                 gcn_drop_rate,
                 gating_reduction,
                 **kwargs
                 ):
        self.output_dim = output_dim
        self.word_dim = word_dim
        self.intermediary_dim1 = intermediary_dim1
        self.gcn_drop_rate = gcn_drop_rate
        self.drop_rate = drop_rate
        self.gating_reduction = gating_reduction

    def __call__(self, inputs, out_matrix, label_feature, is_training, **kwargs):
        with tf.variable_scope("se"):
            self.se = SE(self.drop_rate, self.output_dim, self.gating_reduction, False)
            # 获取多模态融合向量, shape=(batch_size, feat_dim)
            se_output = self.se(inputs, is_training)
            # se_output = tf.layers.batch_normalization(se_output, training=is_training, reuse=False)
        with tf.variable_scope("gcn"):
            self.gcn = GCN(self.word_dim,
                           self.intermediary_dim1,
                           self.output_dim,
                           self.gcn_drop_rate,
                           )
            # shape=(batch_size, num_classes, feat_dim)
            gcn_output = self.gcn(label_feature[0], out_matrix[0], is_training)

        # num_classes = out_matrix.get_shape().as_list()[1]
        # se_output = tf.reshape(tf.tile(se_output, [1, num_classes]), shape=(-1, num_classes, self.output_dim))
        #
        # score = tf.reduce_sum(tf.multiply(gcn_output, se_output), 2)
        score = tf.matmul(se_output, tf.transpose(gcn_output, [1, 0]))
        return score


def run_test():
    model = MLGCN(
        word_dim=768,
        intermediary_dim1=1024,
        output_dim=1024,
        drop_rate=0.6,
        gcn_drop_rate=0.5,
        gating_reduction=8,
    )
    video_embedding = tf.random.normal(shape=(32, 1024))
    text_embedding = tf.random.normal(shape=(32, 768))
    inputs = [video_embedding, text_embedding]
    out_matrix = tf.random.uniform(minval=0.0, maxval=1.0, shape=(32, 82, 82))
    label_feature = tf.random.normal(shape=(32, 82, 768))
    output = model(inputs, out_matrix, label_feature, False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(output)
        print(res.shape)
        print(res)


if __name__ == '__main__':
    run_test()
