import tensorflow as tf
import tensorflow.contrib.slim as slim


class GraphConvolution(object):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def _reset_parameters(self):
        stdv = 1. / tf.sqrt(tf.cast(self.out_features, tf.float32))
        self.weights_params = tf.get_variable("gc_weights", [self.in_features, self.out_features],
                                              initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv))

        if self.bias:
            self.bias_params = tf.get_variable("gc_bias", [self.out_features],
                                               initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv))

    def __call__(self, input, adj):
        self._reset_parameters()
        support = tf.matmul(input, self.weights_params)
        output = tf.matmul(adj, support)
        if self.bias:
            output += self.bias_params

        return output
