import tensorflow.contrib.slim as slim
import tensorflow as tf


class Logistic(object):

    def __init__(self, num_classes, l2_penalty=None):
        self.num_classes = num_classes
        self.l2_penalty = 0.0 if l2_penalty is None else l2_penalty

    def __call__(self, model_input, **kwargs):
        classes_num = model_input.get_shape().as_list()[1]
        feat_num = model_input.get_shape().as_list()[2]
        stdv = 1. / tf.sqrt(tf.cast(feat_num, tf.float32))
        weights = tf.get_variable("logistic_weights", [classes_num, feat_num],
                                  initializer=tf.random_uniform_initializer(minval=-stdv,
                                                                            maxval=stdv))
        bias = tf.get_variable("logistic_bias", [classes_num, 1],
                               initializer=tf.random_uniform_initializer(minval=-stdv,
                                                                         maxval=stdv))

        model_input = model_input * weights
        model_input += bias

        logits = tf.reduce_sum(model_input, 2)

        output = tf.sigmoid(logits)
        output = tf.reshape(output, shape=[-1, self.num_classes])

        return {"predictions": output, "logits": logits}
