import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.model.fusion_head.fusion_se import SE


class MBP(object):

    def __init__(self, drop_rate, hidden1_size, hidden2_size, gating_last_bn=False):
        self.drop_rate = drop_rate
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.gating_last_bn = gating_last_bn
        self.se = SE(drop_rate, hidden1_size=1024, gating_reduction=8)

    def __call__(self, input_list, is_training, **kwargs):
        fusion = self.unit(input_list[0], input_list[1])
        ori = tf.concat(input_list, 1)
        input_tensor = tf.concat([fusion, ori], 1)
        input_tensor = slim.dropout(input_tensor, keep_prob=1.-self.drop_rate, is_training=is_training)
        output = self.se(input_tensor, is_training)
        return output

    def unit(self, x, observed_y):
        x = slim.fully_connected(x, self.hidden1_size, activation_fn=tf.nn.relu,
                                 weights_initializer=slim.variance_scaling_initializer(), scope='x_fc',
                                 reuse=tf.AUTO_REUSE)
        observed_y = slim.fully_connected(observed_y, self.hidden1_size, activation_fn=tf.nn.relu,
                                          weights_initializer=slim.variance_scaling_initializer(), scope='y_fc',
                                          reuse=tf.AUTO_REUSE)
        x = tf.multiply(x, observed_y)

        x = slim.fully_connected(x, self.hidden1_size, activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE,
                                 weights_initializer=slim.variance_scaling_initializer(), scope='fusion_fc')
        return x

    def context_gate(self, x, is_training):
        activation = slim.fully_connected(x, self.hidden2_size, activation_fn=None, scope="cg_hidden_weights",
                                          weights_initializer=slim.variance_scaling_initializer(),
                                          reuse=tf.AUTO_REUSE)
        activation = slim.batch_norm(activation, center=True, scale=True,
                                     is_training=is_training, scope="hidden1_bn", fused=False, reuse=tf.AUTO_REUSE)
        gate_weights = tf.get_variable("gate_weights", [self.hidden2_size, self.hidden2_size],
                                       initializer=slim.variance_scaling_initializer())
        gates = tf.matmul(activation, gate_weights)
        if self.gating_last_bn:
            gates = slim.batch_norm(gates, center=True, scale=True, is_training=is_training, scope="gating_last_bn",
                                    reuse=tf.AUTO_REUSE)
        gates = tf.sigmoid(gates)

        output = activation * gates
        return output


def run_test_model():
    video_embedding = tf.random.normal(shape=(32, 128*128))
    audio_embedding = tf.random.normal(shape=(32, 128*64))
    input_list = [video_embedding, audio_embedding]

    params = {
        'drop_rate': 0.5,
        'hidden1_size': 4096,
        'hidden2_size': 1024,
        'gating_last_bn': True
    }
    model = MBP(drop_rate=0.5, hidden1_size=4096, hidden2_size=1024, gating_last_bn=True)

    output = model(input_list, is_training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)
        print(output.shape)


if __name__ == '__main__':
    run_test_model()
