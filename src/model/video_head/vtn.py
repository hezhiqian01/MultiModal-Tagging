from src.model.text_head.bert_base import transformer_model, embedding_postprocessor, create_attention_mask_from_input_mask
import tensorflow as tf
import tensorflow.contrib.slim as slim


class VTN(object):

    def __init__(self, hidden_size=768, max_frames=300, **kwargs):
        self.hidden_size = hidden_size
        self.max_frames = max_frames

    def __call__(self, input_tensor, is_training, mask, **kwargs):
        """
        :param input_tensor: video或者audio特征，shape=(batch_size, max_frames, embedding_size)
        :param is_training:
        :param mask: shape=(32, 300)
        :param kwargs:
        :return:
        """
        embedding_size = input_tensor.get_shape().as_list()[-1]
        cls_embedding = tf.get_variable('cls_embedding', shape=(1, 1, embedding_size),
                                        # initializer=tf.truncated_normal_initializer(stddev=0.02))
                                        initializer=tf.initializers.zeros)

        cls_embedding = tf.tile(cls_embedding, multiples=[tf.shape(input_tensor)[0], 1, 1])
        embeddings = tf.concat([cls_embedding, input_tensor], axis=1)
        # print(embeddings.shape)
        # token_type_ids = tf.reshape(tf.Variable(cls_embedding + [0]*128 + [1] * 64 + [2] * 128), shape=(-1, 1))
        # # token_type_ids = tf.tile(token_type_ids, )
        # print(token_type_ids.shape)
        embeddings = embedding_postprocessor(input_tensor, use_token_type=False, token_type_ids=None)
        print(embeddings.shape)
        # mask = tf.concat([tf.ones(shape=(tf.shape(input_tensor)[0], 1)), mask], axis=1)
        attention_mask = create_attention_mask_from_input_mask(mask, mask)
        output = transformer_model(
            input_tensor=embeddings,
            attention_mask=attention_mask,
            hidden_size=768,
            num_hidden_layers=3,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            do_return_all_layers=False,
            reuse_variables=None)
        output = output[:, 0, :]
        # output = tf.reduce_mean(output, axis=1)
        # output = tf.layers.dropout(output, rate=0.8, training=is_training)
        return output


def run_test_model():
    video_tensor = tf.random.normal(shape=(32, 300, 768))
    audio_tensor = tf.random.normal(shape=(32, 300, 128))
    mask = tf.ones(shape=(32, 300))
    model = VTN()
    output = model(input_tensor=video_tensor, is_training=True, mask=mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)
        print(output.shape)


if __name__ == '__main__':
    run_test_model()
