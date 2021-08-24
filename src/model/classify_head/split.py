import tensorflow as tf
import tensorflow.contrib.slim as slim


class SplitMLP(object):

    def __init__(self, hierarchical_depth, hierarchy_classes, num_classes, **unused):
        self.init = slim.variance_scaling_initializer()
        self.hierarchical_depth = hierarchical_depth
        self.hierarchical_class = hierarchy_classes
        self.num_classes = num_classes
        self.drop_rate = 0.9
        self.l2_penalty = 0.0

    def __call__(self, embedding, is_training, **kwargs):
        global_hidden = slim.fully_connected(
            embedding, self.num_classes, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.l2_penalty),
            biases_regularizer=slim.l2_regularizer(self.l2_penalty),
            weights_initializer=slim.variance_scaling_initializer(), scope='global_preds')

        global_preds = tf.nn.sigmoid(global_hidden)

        local_preds = []
        for i, depth in enumerate(self.hierarchical_depth):
            with tf.variable_scope('local-{}'.format(i)):
                local_pred = self.unit(embedding, depth, class_num=self.hierarchical_class[i],
                                       drop_rate=self.drop_rate, is_training=is_training)
                local_preds.append(local_pred)
        local_preds = tf.concat(local_preds, axis=1)

        predictions = 0.5 * local_preds + 0.5 * global_preds
        res = {
            'global_predictions': global_preds,
            'local_predictions': local_preds,
            'predictions': predictions
        }
        return res

    def unit(self, x, hidden_size, drop_rate, class_num, is_training):
        x = slim.fully_connected(x, num_outputs=hidden_size, activation_fn=tf.nn.relu,
                                 weights_initializer=self.init)
        x = slim.batch_norm(x, is_training=is_training)
        x = slim.dropout(x, keep_prob=1.-drop_rate, is_training=is_training)
        x = slim.fully_connected(
            x, class_num, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.l2_penalty),
            biases_regularizer=slim.l2_regularizer(self.l2_penalty),
            weights_initializer=slim.variance_scaling_initializer(), scope='local_preds')
        preds = tf.nn.sigmoid(x)
        return preds
