import tensorflow as tf


class Sigmoid(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, model_input, **kwargs):
        output = tf.sigmoid(model_input)
        return {"predictions": output, "logits": model_input}

