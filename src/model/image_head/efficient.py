from src.model.image_head.efficientNet.efficientnet_builder import build_model, build_model_base
import tensorflow as tf


images = tf.random.uniform(minval=0, maxval=255, shape=(32, 224, 224, 3))
output = build_model_base(images, 'efficientnet-b0', is_training=False)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    features, endpoints = sess.run(output)
    print(type(features), features.shape)

    print(type(endpoints))
    print(endpoints.keys())


