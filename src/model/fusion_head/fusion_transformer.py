from src.model.text_head.bert_base import transformer_model, embedding_postprocessor
import tensorflow as tf
import tensorflow.contrib.slim as slim


class FusionTrm(object):

    def __init__(self, hidden_size=768, batch_size=32):
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def __call__(self, input_list, is_training, **kwargs):
        """
        :param input_list: 包含video(128, 128)，audio(64, 128)，text(128, 384)的特征
        :param is_training:
        :param kwargs:
        :return:
        """
        video_tensor, audio_tensor, text_tensor = input_list
        video_size = video_tensor.get_shape().as_list()[-1]
        audio_size = audio_tensor.get_shape().as_list()[-1]
        text_size = text_tensor.get_shape().as_list()[-1]

        video_weights = tf.get_variable('video_weights', shape=(video_size, self.hidden_size),
                                        initializer=slim.variance_scaling_initializer())
        video_embedding = tf.matmul(video_tensor, video_weights)

        audio_weights = tf.get_variable('audio_weights', shape=(audio_size, self.hidden_size),
                                        initializer=slim.variance_scaling_initializer())
        audio_embedding = tf.matmul(audio_tensor, audio_weights)

        text_weights = tf.get_variable('text_weights', shape=(text_size, self.hidden_size),
                                       initializer=slim.variance_scaling_initializer())
        text_embedding = tf.matmul(text_tensor, text_weights)

        cls_embedding = tf.get_variable('cls_embedding', shape=(1, 1, self.hidden_size),
                                        initializer=slim.variance_scaling_initializer())
        cls_embedding = tf.tile(cls_embedding, multiples=[self.batch_size, 1, 1])

        sep_embedding = tf.get_variable('sep_embedding', shape=(1, 1, self.hidden_size),
                                        initializer=slim.variance_scaling_initializer())
        sep_embedding = tf.tile(sep_embedding, multiples=[self.batch_size, 1, 1])

        embeddings = tf.concat([cls_embedding, video_embedding, sep_embedding, audio_embedding, sep_embedding, text_embedding], axis=1)
        print(embeddings.shape)
        # token_type_ids = tf.reshape(tf.Variable(cls_embedding + [0]*128 + [1] * 64 + [2] * 128), shape=(-1, 1))
        # # token_type_ids = tf.tile(token_type_ids, )
        # print(token_type_ids.shape)
        embeddings = embedding_postprocessor(embeddings, use_token_type=False, token_type_ids=None)

        output = transformer_model(
            input_tensor=embeddings,
            attention_mask=None,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            do_return_all_layers=False,
            reuse_variables=None)
        return output[:, 0, :]


def run_test_model():
    video_tensor = tf.random.normal(shape=(32, 128, 128))
    audio_tensor = tf.random.normal(shape=(32, 64, 128))
    text_tensor = tf.random.normal(shape=(32, 128, 384))

    model = FusionTrm()
    output = model(input_list=[video_tensor, audio_tensor, text_tensor], is_training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)
        print(output.shape)


if __name__ == '__main__':
    run_test_model()
