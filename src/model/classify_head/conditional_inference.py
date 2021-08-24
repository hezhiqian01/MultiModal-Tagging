import tensorflow.contrib.slim as slim
import tensorflow as tf


class ConditionalInference(object):

    def __init__(self, num_classes, hidden1_size, hidden2_size, cg_hidden_size, steps, gating_last_bn=True, **kwargs):
        self.vocab_size = num_classes
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.cg_hidden_size = cg_hidden_size
        self.gating_last_bn = gating_last_bn
        self.steps = steps

    def forward(self, model_input, is_training):
        observed_y = tf.ones(shape=(1, self.vocab_size))

        with tf.variable_scope("conditional_inference", reuse=tf.AUTO_REUSE):
            # first step
            y_predict = self.unit(model_input, observed_y, is_training)
            # 找到预测的分数的最大值的索引，并且不能算已经发现的最大值的索引
            y_predict_max_score_idx = tf.argmax(y_predict, axis=1)
            observed_y = tf.zeros(shape=(1, self.vocab_size))
            observed_y += tf.one_hot(y_predict_max_score_idx, depth=self.vocab_size)

            for _ in range(self.steps):
                y_predict = self.unit(model_input, observed_y, is_training)
                y_predict_max_score_idx = tf.argmax(y_predict - observed_y, axis=1)
                observed_y += tf.one_hot(y_predict_max_score_idx, depth=self.vocab_size)

            return y_predict

    def __call__(self, model_input, is_training=True, **kwargs):
        prediction = self.forward(model_input, is_training)
        return {"predictions": prediction}

    def unit(self, x, observed_y, is_training):
        x = slim.fully_connected(x, self.hidden1_size, activation_fn=tf.nn.relu,
                                 weights_initializer=slim.variance_scaling_initializer(), scope='x_fc',
                                 reuse=tf.AUTO_REUSE)
        observed_y = slim.fully_connected(observed_y, self.hidden1_size, activation_fn=tf.nn.relu,
                                          weights_initializer=slim.variance_scaling_initializer(), scope='y_fc',
                                          reuse=tf.AUTO_REUSE)
        x = tf.multiply(x, observed_y)

        x = slim.fully_connected(x, self.hidden2_size, activation_fn=tf.nn.relu, reuse=tf.AUTO_REUSE,
                                 weights_initializer=slim.variance_scaling_initializer(), scope='fusion_fc')
        x = self.context_gate(x, is_training)
        x = slim.fully_connected(x, self.vocab_size, activation_fn=None, scope='final_fc', reuse=tf.AUTO_REUSE,
                                 weights_initializer=slim.variance_scaling_initializer())
        x = tf.nn.sigmoid(x)

        return x

    def context_gate(self, x, is_training):
        activation = slim.fully_connected(x, self.cg_hidden_size, activation_fn=None, scope="cg_hidden_weights",
                                          weights_initializer=slim.variance_scaling_initializer(),
                                          reuse=tf.AUTO_REUSE)
        activation = slim.batch_norm(activation, center=True, scale=True,
                                     is_training=is_training, scope="hidden1_bn", fused=False, reuse=tf.AUTO_REUSE)
        gate_weights = tf.get_variable("gate_weights", [self.cg_hidden_size, self.cg_hidden_size],
                                       initializer=slim.variance_scaling_initializer())
        gates = tf.matmul(activation, gate_weights)
        if self.gating_last_bn:
            gates = slim.batch_norm(gates, center=True, scale=True, is_training=is_training, scope="gating_last_bn",
                                    reuse=tf.AUTO_REUSE)
        gates = tf.sigmoid(gates)

        output = activation * gates
        return output


def run_test_model():
    model = ConditionalInference(num_classes=10, hidden1_size=8, hidden2_size=1024, cg_hidden_size=1024, steps=5)
    model_input = tf.Variable(tf.random.normal(shape=(32, 128)))
    output = model(model_input, is_training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)
        print(output['predictions'].shape)
        print(output)


if __name__ == '__main__':
    run_test_model()
