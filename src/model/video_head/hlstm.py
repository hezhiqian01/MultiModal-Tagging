import tensorflow as tf


class HLSTM(object):

    def __init__(self, max_frames, lstm_size=1024, lstm_layers=1, num_inputs_to_lstm=20):
        self.max_num_frames = max_frames
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.num_inputs_to_lstm = num_inputs_to_lstm

    def __call__(self, model_input, num_frames, **kwargs):
        """Creates a model which uses a stack of LSTMs to represent the video.
           Args:
             model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                          input features.
             num_frames: A vector of length 'batch' which indicates the number of
                  frames for each video (before padding).
           Returns:
             A dictionary with a tensor containing the probability predictions of the
             model in the 'predictions' key. The dimensions of the tensor are
             'batch_size' x 'num_classes'.
           """
        print(model_input.shape)

        L1_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(self.lstm_layers)
            ],
            state_is_tuple=False)

        L2_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    self.lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(self.lstm_layers)
            ],
            state_is_tuple=False)

        split_model_input = tf.split(model_input, self.num_inputs_to_lstm, axis=1, name='lstm_l1_split')
        len_lower_lstm = self.max_num_frames // self.num_inputs_to_lstm

        num_frames_L1 = [tf.minimum(len_lower_lstm, tf.maximum(0, num_frames - int(len_lower_lstm) * i)) for i in
                         range(self.num_inputs_to_lstm)]

        L1_outputs = []
        for i in range(self.num_inputs_to_lstm):
            with tf.variable_scope("RNN_L1") as scope:
                if i > 0:
                    scope.reuse_variables()
                _, state = tf.nn.dynamic_rnn(L1_stacked_lstm, split_model_input[i],
                                             sequence_length=num_frames_L1[i],
                                             dtype=tf.float32)
                L1_outputs.append(state)

        L2_input = tf.stack(L1_outputs, axis=1)

        with tf.variable_scope("RNN_L2"):
            _, state = tf.nn.dynamic_rnn(L2_stacked_lstm, L2_input,
                                         sequence_length=tf.cast(
                                             tf.ceil(tf.cast(num_frames, tf.float32) / len_lower_lstm), tf.int32),
                                         dtype=tf.float32)
        return state


def run_test_model():
    model = HLSTM(max_frames=300)

    model_input = tf.random.normal(shape=(32, 300, 1024))
    num_frames = tf.random.uniform(shape=(32, ), minval=100, maxval=200, dtype=tf.int32)
    print(num_frames)
    #
    output = model(model_input, num_frames)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)
        print(output)
        print(output.shape)


if __name__ == '__main__':
    run_test_model()

