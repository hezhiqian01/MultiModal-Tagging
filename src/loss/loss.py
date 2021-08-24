"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf
import numpy as np
import pandas as pd


class BaseLoss(object):
    """Inherit from this class when implementing new losses."""
    def __init__(self, *args, **kwargs):
        pass

    def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
        """Calculates the average loss of the examples in a mini-batch.
         Args:
          unused_predictions: a 2-d tensor storing the prediction scores, in which
            each row represents a sample in the mini-batch and each column
            represents a class.
          unused_labels: a 2-d tensor storing the labels, which has the same shape
            as the unused_predictions. The labels must be in the range of 0 and 1.
          unused_params: loss specific parameters.
        Returns:
          A scalar loss tensor.
        """
        raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
    """Calculate the cross entropy loss between the predictions and labels.
    """

    def calculate_loss(self, predictions, labels, **unused_params):
        with tf.name_scope("loss_xent"):
            epsilon = 1e-8
            label_smooth_rate = unused_params.get('label_smooth_rate', 0.0)
            float_labels = tf.cast(labels, tf.float32) * (1.0 - label_smooth_rate) + \
                           (1.0 - tf.cast(labels, tf.float32)) * label_smooth_rate

            cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                    1 - float_labels) * tf.log(1 - predictions + epsilon)
            cross_entropy_loss = tf.negative(cross_entropy_loss)
            alpha = unused_params.get('loss_weight', 1.0)  # alpha shape=[batch_size]
            return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1) * alpha)


class HingeLoss(BaseLoss):
    """Calculate the hinge loss between the predictions and labels.
    Note the subgradient is used in the backpropagation, and thus the optimization
    may converge slower. The predictions trained by the hinge loss are between -1
    and +1.
    """

    def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
        with tf.name_scope("loss_hinge"):
            float_labels = tf.cast(labels, tf.float32)
            all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
            all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
            sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
            hinge_loss = tf.maximum(
                all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
            return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class SoftmaxLoss(BaseLoss):
    """Calculate the softmax loss between the predictions and labels.
    The function calculates the loss in the following way: first we feed the
    predictions to the softmax activation function and then we calculate
    the minus linear dot product between the logged softmax activations and the
    normalized ground truth label.
    It is an extension to the one-hot label. It allows for more than one positive
    labels for each sample.
    """

    def calculate_loss(self, predictions, labels, **unused_params):
        with tf.name_scope("loss_softmax"):
            epsilon = 1e-8
            float_labels = tf.cast(labels, tf.float32)
            # l1 normalization (labels are no less than 0)
            label_rowsum = tf.maximum(
                tf.reduce_sum(float_labels, 1, keep_dims=True),
                epsilon)
            norm_float_labels = tf.div(float_labels, label_rowsum)
            softmax_outputs = tf.nn.softmax(predictions)
            softmax_loss = tf.negative(tf.reduce_sum(
                tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
        return tf.reduce_mean(softmax_loss)


class HMCNLoss(BaseLoss):

    def calculate_loss(self, predictions, labels, **unused_params):

        local_predictions = unused_params.get('local_predictions')
        global_predictions = unused_params.get('global_predictions')

        ce_loss = CrossEntropyLoss()
        with tf.variable_scope("local-loss"):
            local_loss = ce_loss.calculate_loss(local_predictions, labels, **unused_params)
        with tf.variable_scope('global-loss'):
            global_loss = ce_loss.calculate_loss(global_predictions, labels, **unused_params)

        with tf.variable_scope('diff-loss'):
            diff_loss = tf.reduce_sum(tf.pow(global_predictions - local_predictions, 2), axis=1)
            diff_loss = tf.reduce_mean(diff_loss)

        loss = global_loss + local_loss + 1. * diff_loss
        return loss


class DBLoss(BaseLoss):

    def __init__(self,
                 freq_file,
                 focal=dict(
                     focal=True,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 ):
        super(DBLoss, self).__init__()
        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        self.class_freq = np.asarray(pd.read_pickle(freq_file)['class_freq'])
        self.neg_class_freq = np.asarray(pd.read_pickle(freq_file)['neg_class_freq'])
        self.num_classes = self.class_freq.shape[0]
        # train_num 等于训练集总数, N
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]

        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = tf.cast(- tf.log(
            self.train_num / self.class_freq - 1) * init_bias, tf.float32)

        self.freq_inv = tf.ones(self.class_freq.shape) / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def calculate_loss(self, predictions, labels, **unused_params):
        logits = unused_params.get('logits')
        assert logits is not None

        weight = self.reweight_functions(labels)
        # print(weight)

        logits, weight = self.logit_reg_functions(tf.cast(labels, tf.float32), logits, weight)

        predictions = tf.nn.sigmoid(logits)

        if self.focal:
            logpt = -self.ce_loss(predictions, labels)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = tf.exp(logpt)
            loss = self.ce_loss(
                predictions, labels, loss_weight=weight)

            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.ce_loss(predictions, labels, loss_weight=weight, reduction=None)

        loss_weight = unused_params.get('loss_weight', 1.0)
        loss = loss_weight * loss

        loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
        return loss

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale + logits * labels
            weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def reweight_functions(self, labels):
        labels = tf.cast(labels, tf.float32)
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(labels)
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(labels)
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(labels)
        else:
            raise NotImplementedError

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = tf.maximum(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / tf.maximum(weight)

        return weight

    def RW_weight(self, labels):
        raise NotImplementedError

    def CB_weight(self, labels):
        raise NotImplementedError

    def rebalance_weight(self, gt_labels):
        repeat_rate = tf.reduce_sum(gt_labels * self.freq_inv, axis=1, keepdims=True)
        pos_weight = tf.expand_dims(self.freq_inv, axis=0) / repeat_rate
        # pos and neg are equally treated
        weight = tf.nn.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def ce_loss(self, predictions, labels, **unused_params):
        epsilon = 1e-8
        label_smooth_rate = unused_params.get('label_smooth_rate', 0.0)
        float_labels = tf.cast(labels, tf.float32) * (1.0 - label_smooth_rate) + \
                       (1.0 - tf.cast(labels, tf.float32)) * label_smooth_rate

        cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                1 - float_labels) * tf.log(1 - predictions + epsilon)
        cross_entropy_loss = tf.negative(cross_entropy_loss)
        alpha = unused_params.get('loss_weight', 1.0)  # alpha shape=[batch_size]
        reduction = unused_params.get('reduction', None)
        if reduction is None:
            return cross_entropy_loss * alpha
        elif reduction == 'mean':
            return tf.reduce_mean(cross_entropy_loss, 1) * alpha
        else:
            raise NotImplementedError


def run_test_db_loss():
    class_freq = {
        'class_freq': np.array([10, 2, 3]),
        'neg_class_freq': np.array([5, 13, 12])
    }
    pd.to_pickle(class_freq, './class_freq.pkl')
    class_freq = pd.read_pickle('./class_freq.pkl')
    print(class_freq)

    loss_model = DBLoss(
        focal=dict(
            focal=True,
            balance_param=2.0,
            gamma=2,
        ),
        map_param=dict(
            alpha=0.1,
            beta=10,
            gamma=0.3
        ),
        logit_reg=dict(
            neg_scale=5.0,
            init_bias=0.05
        ),
        reweight_func='rebalance',  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
        weight_norm=None,  # None, 'by_instance', 'by_batch'
        freq_file='./class_freq.pkl')

    logits = tf.constant([[2.1, 1.3, 4.5], [-5.64, 3.556, 3.3], [4.5, 4.5, 5.1]])
    labels = tf.constant([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    predictions = tf.nn.sigmoid(logits)

    loss = loss_model.calculate_loss(predictions, labels, logits=logits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss = sess.run(loss)
        print(loss)
        # print(loss.shape)


if __name__ == '__main__':
    run_test_db_loss()

