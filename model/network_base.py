
import numpy as np
import tensorflow as tf


class Network(object):
    def __init__(self, num_classes, lrn_rate_placeholder, wd_rate_placeholder, wd_rate_placeholder2,
                 mode='train', initializer='he', fix_blocks=0,
                 RV=False, fine_tune_filename=None,
                 wd_mode=0, optimizer='mom', momentum=0.9, fisher_filename=None,
                 gpu_num=1, fisher_epsilon=0, data_format='NHWC',
                 float_type=tf.float32, separate_regularization=False):
        self.mode = mode
        self.global_step = None
        self._extra_train_ops = []
        self._extra_loss = []
        self.fix_blocks = fix_blocks
        self.num_classes = num_classes
        self.fine_tune_filename = fine_tune_filename
        self.wd_mode = wd_mode
        self.optimizer = optimizer
        self.momentum = momentum
        self.lrn_rate_placeholder = lrn_rate_placeholder
        self.wd_rate_placeholder = wd_rate_placeholder
        self.wd_rate_placeholder2 = wd_rate_placeholder2
        self.fisher_filename = fisher_filename
        self.gpu_num = gpu_num
        self.fisher_epsilon = fisher_epsilon

        assert data_format in ['NCHW', 'NHWC']
        self.data_format = data_format

        self.RV = RV
        self.net_activations = dict()
        self.net_activations_sorted_keys = []
        self.initializer = initializer

        # TODO: Problem is that tf can not import a pre-trained with different data type.
        # TODO: So this is not implemented yet, all using tf.float32 at this moment.
        self.float_type = float_type

        self.new_layers_names = ['logits', 'global_step']
        self.sp_group = self.new_layers_names

    def inference(self, images):
        self.logits = None
        raise NotImplementedError("Implement this method.")

    def compute_loss(self, labels, logits=None):
        self.labels = labels
        if logits is None:
            logits = self.logits

        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            self.loss = tf.reduce_mean(xent, name='xent')
            self.wd = tf.convert_to_tensor(0.0, dtype=self.float_type)
            if self.mode == 'train':
                self.wd = self._decay(self.wd_mode)
            cost = self.loss + self.wd

        return cost

    def _default_train_op(self, labels, logits=None):
        # regularizers are in the loss and computed into momentum, like always.
        self.cost = self.compute_loss(labels, logits)

        if self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate_placeholder)
        elif self.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate_placeholder, self.momentum)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate_placeholder, self.momentum)
        else:
            print 'unknown optimizer name: ', self.optimizer
            print 'Default to Momentum.'
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate_placeholder, self.momentum)

        grads_vars = optimizer.compute_gradients(self.cost, colocate_gradients_with_ops=True)
        apply_op = optimizer.apply_gradients(
            grads_vars,
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        return self.train_op

    def build_train_op(self, labels, logits=None):
        """
        :param labels:
        :param logtis:
        :return:
        """
        return self._default_train_op(labels, logits=None)

    def _decay(self, mode):
        """L2 weight decay loss."""
        print '================== weight decay info   ===================='
        if mode == 0:
            print 'Applying L2 regularization...'
            losses_existing_layers = 0.0
            losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.sp_group):
                        losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    losses_existing_layers += tf.nn.l2_loss(v)
        elif mode == 1:
            print('Applying L2-SP regularization... with exception of', self.sp_group)
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            losses_existing_layers = 0.0
            losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.sp_group):
                        losses_new_layers += tf.nn.l2_loss(v)
                        print('layers with L2 :', v.name)
                        continue

                    name = v.name.split(':')[0]
                    if reader.has_tensor(name):
                        pre_trained_weights = reader.get_tensor(name)
                        print('layers with L2-SP :', v.name)
                    else:
                        raise KeyError('not find %s' % name)

                    losses_existing_layers += tf.nn.l2_loss(v - pre_trained_weights)
        else:
            print('No regularization...')
            return tf.convert_to_tensor(0.0)
        return tf.add(tf.multiply(self.wd_rate_placeholder, losses_existing_layers),
                      tf.multiply(self.wd_rate_placeholder2, losses_new_layers),
                      name='weight_decay')

