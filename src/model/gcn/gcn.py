import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.model.gcn.layers import GraphConvolution
import numpy as np


class GCN(object):

    def __init__(self, nfeat, nhid, output_dim, drop_rate, **kwargs):
        self.nfeat = nfeat
        self.nhid = nhid
        self.output_dim = output_dim
        self.drop_rate = drop_rate

    def __call__(self, x, adj, is_training=True):
        x = tf.layers.batch_normalization(x, training=is_training, reuse=tf.AUTO_REUSE)

        A = adj
        adj = self.gen_adj(A)

        with tf.variable_scope("gc1"):
            self.gc1 = GraphConvolution(self.nfeat, self.nhid)
            x = tf.nn.leaky_relu(self.gc1(x, adj))
            x = slim.dropout(x, keep_prob=1. - self.drop_rate, is_training=is_training,
                             scope="gcn_dropout")
        with tf.variable_scope("gc2"):
            self.gc2 = GraphConvolution(self.nhid, self.output_dim)
            x = self.gc2(x, adj)
            # x = tf.layers.batch_normalization(x, training=is_training, reuse=tf.AUTO_REUSE)
            # x = tf.nn.leaky_relu(self.gc2(x, adj))
            # x = slim.dropout(x, keep_prob=1. - self.drop_rate, is_training=is_training,
            #                  scope="gcn_dropout")
            return x

    def gen_adj(self, A):
        D = tf.pow(tf.reduce_sum(A, 1), -0.5)
        D = tf.matrix_diag(D)
        adj = tf.matmul(tf.transpose(tf.matmul(A, D), [1, 0]), D)
        return adj


def run_test():
    import numpy as np
    params = {
        'nfeat': 768,
        'nhid': 1024,
        'output_dim': 1024,
        'drop_rate': 0.4
    }
    x = tf.random.normal(shape=(32, 82, 768))
    adj = tf.cast(np.load("/Users/zhiqianhe/MyProjects/腾讯广告算法/txad/dataset/co_occurrence.npy"), tf.float32)
    adj = tf.reshape(tf.tile(adj, [32, 1]), shape=(32, 82, 82))
    # print(adj)
    model = GCN(**params)

    output = model(x, adj)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(output)
        print(res.shape)
        print(res)


if __name__ == '__main__':
    run_test()
