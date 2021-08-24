import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class MLP(object):

    def __init__(self, num_classes, hidden_size=1024, mlp_num=2, drop_rate=0.3, l2_penalty=1e-8):
        self.num_classes = num_classes
        self.l2_penalty = l2_penalty
        self.hidden_size = hidden_size
        self.mlp_num = mlp_num
        self.drop_rate = drop_rate

    def __call__(self, model_input, is_training, **kwargs):
        x = model_input
        for i in range(self.mlp_num):
            with tf.variable_scope("mlp-{}".format(i+1)):
                # x = slim.fully_connected(x, self.hidden_size, activation_fn=tf.nn.relu,
                #                          weights_initializer=slim.variance_scaling_initializer(),
                #                          )
                x = tf.layers.dense(x, units=self.hidden_size, activation=gelu)
                # x = slim.dropout(x, keep_prob=1.-self.drop_rate, is_training=is_training)
                x = tf.layers.dropout(x, rate=self.drop_rate, training=is_training)
        logits = slim.fully_connected(
            x, self.num_classes, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.l2_penalty),
            biases_regularizer=slim.l2_regularizer(self.l2_penalty),
            weights_initializer=slim.variance_scaling_initializer())
        output = tf.nn.sigmoid(logits)
        return {"predictions": output, "logits": logits}


def run_test_model():
    x = tf.random.normal(shape=(32, 1024))

    model = MLP(num_classes=82, l2_penalty=1e-8)

    output = model(x, is_training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)

        print(output)


if __name__ == '__main__':
    run_test_model()

