import tensorflow as tf


class AttentionClusterModule(object):

    def __init__(self, feature_size, max_frames, dropout_rate, cluster_size,
                 add_batch_norm, shift_operation, **unused_params):
        """ Initialize AttentionClusterModule.
        :param feature_size: int
        :param max_frames: vector of int
        :param dropout_rate: float
        :param cluster_size: int
        :param add_batch_norm: bool
        :param shift_operation: bool
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.add_batch_norm = add_batch_norm
        self.dropout_rate = dropout_rate
        self.shift_operation = shift_operation
        self.cluster_size = cluster_size

    def __call__(self, inputs, is_training, **unused_params):
        """ Forward method for AttentionClusterModule.
        :param inputs: 3D Tensor of size 'batch_size x max_frames x feature_size'
        :return: 2D Tensor of size 'batch_size x (feature_size * cluster_size)
        """
        inputs = tf.reshape(inputs, [-1, self.feature_size])
        reshaped_inputs = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])

        attention_weights = tf.layers.dense(inputs, self.cluster_size, use_bias=False, activation=None)
        float_cpy = tf.cast(self.feature_size, dtype=tf.float32)
        attention_weights = tf.divide(attention_weights, tf.sqrt(float_cpy))
        if self.add_batch_norm:
            attention_weights = tf.layers.batch_normalization(attention_weights, training=is_training)
        if is_training:
            attention_weights = tf.nn.dropout(attention_weights, self.dropout_rate)
        attention_weights = tf.nn.softmax(attention_weights)

        reshaped_attention = tf.reshape(attention_weights, [-1, self.max_frames, self.cluster_size])
        transposed_attention = tf.transpose(reshaped_attention, perm=[0, 2, 1])
        # -> transposed_attention: batch_size x cluster_size x max_frames
        activation = tf.matmul(transposed_attention, reshaped_inputs)
        # -> activation: batch_size x cluster_size x feature_size
        transformed_activation = tf.transpose(activation, perm=[0, 2, 1])
        # -> transformed_activation: batch_size x feature_size x cluster_size
        transformed_activation = tf.nn.l2_normalize(transformed_activation, 1)

        if self.shift_operation:
            alpha = tf.get_variable("alpha",
                                    [self.cluster_size],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable("beta",
                                   [self.cluster_size],
                                   initializer=tf.constant_initializer(0.0))
            transformed_activation = tf.multiply(transformed_activation, alpha)
            transformed_activation = tf.add(transformed_activation, beta)

        normalized_activation = tf.nn.l2_normalize(transformed_activation, 1)
        normalized_activation = tf.reshape(normalized_activation, [-1, self.cluster_size * self.feature_size])
        normalized_activation = tf.nn.l2_normalize(normalized_activation)

        return normalized_activation


def run_test_model():
    model_input = tf.random.normal(shape=(32, 300, 128))
    model = AttentionClusterModule(feature_size=128, max_frames=300, dropout_rate=0.2, cluster_size=128,
                                   add_batch_norm=True, shift_operation=True)

    output = model(model_input, is_training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)
        print(output)
        print(output.shape)


if __name__ == '__main__':
    run_test_model()
