import tensorflow as tf
import tensorflow.contrib.slim as slim


class HMCN(object):

    def __init__(self, hierarchical_depth, hierarchy_classes, global2local, num_classes, **unused):
        self.hierarchical_depth = hierarchical_depth
        self.hierarchical_class = hierarchy_classes
        self.global2local = global2local
        self.classes_num = num_classes
        # self.init = tf.initializers.random_normal(stddev=0.1)
        self.init = slim.variance_scaling_initializer()

    def global_layer(self, x, depth, is_training):
        x = slim.fully_connected(x, num_outputs=depth, activation_fn=tf.nn.relu,
                                 weights_initializer=self.init)
        x = slim.batch_norm(x, is_training=is_training)
        x = slim.dropout(x, keep_prob=0.5, is_training=is_training)
        return x

    def local_layer(self, x, depth, class_num, is_training):
        x = slim.fully_connected(x, num_outputs=depth, activation_fn=tf.nn.relu,
                                 weights_initializer=self.init)
        x = slim.batch_norm(x, is_training=is_training)
        x = slim.fully_connected(x, num_outputs=class_num,
                                 weights_initializer=self.init)
        return x

    def linear(self, x):
        x = slim.fully_connected(x, self.classes_num, activation_fn=None, biases_initializer=None,
                                 weights_initializer=self.init)
        return x

    def __call__(self, embedding, is_training, **kwargs):
        local_layer_outputs = []
        global_layer_activation = embedding

        for i in range(1, len(self.hierarchical_depth)):
            with tf.variable_scope("hmcn-{}".format(i)):
                local_layer_activation = self.global_layer(x=global_layer_activation, depth=self.hierarchical_depth[i],
                                                           is_training=is_training)
                local_layer_outputs.append(self.local_layer(local_layer_activation, self.global2local[i],
                                                            self.hierarchical_class[i-1], is_training))
                if i < len(self.hierarchical_depth) - 1:
                    global_layer_activation = tf.concat([local_layer_activation, embedding], 1)
                else:
                    global_layer_activation = local_layer_activation

        with tf.variable_scope("hmcn-global-linear"):
            global_layer_output = self.linear(global_layer_activation)

        local_layer_output = tf.concat(local_layer_outputs, 1)

        global_logits, local_logits, logits = global_layer_output, local_layer_output, 0.5 * global_layer_output + 0.5 * local_layer_output

        global_preds = tf.nn.sigmoid(global_logits)
        local_preds = tf.nn.sigmoid(local_logits)
        predictions = tf.nn.sigmoid(logits)

        res = {
            'global_predictions': global_preds,
            'local_predictions': local_preds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'logits': logits,
            'predictions': predictions
        }
        return res


def run_test_model():
    embedding = tf.random.normal(shape=(32, 16384+2048))
    hierarchical_depth = [0, 384, 384, 384, 384, 384, 384]
    hierarchy_classes = [9, 23, 11, 6, 10, 23]
    global2local = [0, 4, 55, 43, 30, 30, 1]
    classes_num = 82
    model = HMCN(hierarchical_depth, hierarchy_classes, global2local, classes_num)

    output = model(embedding, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output)
        print(output)


if __name__ == '__main__':
    run_test_model()

