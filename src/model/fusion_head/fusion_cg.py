import tensorflow.contrib.slim as slim
import tensorflow as tf


class ContextGating(object):

    def __init__(self, drop_rate, hidden1_size, gating_last_bn=True, **kwargs):
        self.drop_rate = drop_rate
        self.hidden_size = hidden1_size
        self.gating_last_bn = gating_last_bn

    def __call__(self, input_list, is_training, **kwargs):
        concat_feat = tf.concat(input_list, 1)
        if self.drop_rate > 0.:
            concat_feat = slim.dropout(concat_feat, keep_prob=1. - self.drop_rate, is_training=is_training,
                                       scope="concat_feat_dropout")
        activation = slim.fully_connected(concat_feat, self.hidden_size, activation_fn=None, scope="hidden_weights",
                                          weights_initializer=slim.variance_scaling_initializer())
        activation = slim.batch_norm(activation, center=True, scale=True,
                                     is_training=is_training, scope="hidden1_bn", fused=False)
        gate_weights = tf.get_variable("gate_weights", [self.hidden_size, self.hidden_size],
                                       initializer=slim.variance_scaling_initializer())
        gates = tf.matmul(activation, gate_weights)
        if self.gating_last_bn:
            gates = slim.batch_norm(gates, center=True, scale=True, is_training=is_training, scope="gating_last_bn")
        gates = tf.sigmoid(gates)

        output = activation * gates
        return output
