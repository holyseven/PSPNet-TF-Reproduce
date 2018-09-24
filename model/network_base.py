from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf


class Network(object):
    def __init__(self, num_classes,
                 mode='train', initializer='he', fine_tune_filename=None,
                 wd_mode=0, optimizer='mom', momentum=0.9,
                 gpu_num=1, data_format='NHWC',
                 float_type=tf.float32):
        self.mode = mode
        self.global_step = None
        self._extra_train_ops = []
        self._extra_loss = []
        self.num_classes = num_classes
        self.fine_tune_filename = fine_tune_filename
        self.wd_mode = wd_mode
        self.optimizer = optimizer
        self.momentum = momentum
        self.gpu_num = gpu_num

        assert data_format in ['NCHW', 'NHWC']
        self.data_format = data_format
        self.initializer = initializer

        # TODO: Problem is that tf can not import a pre-trained with different data type.
        # TODO: So this is not implemented yet, all using tf.float32 at this moment.
        self.float_type = float_type

        self.new_layers_names = ['logits', 'global_step']
        self.sp_group = self.new_layers_names
        self.logits = None

        # < initialize some placeholders >
        self.wd_rate_ph = tf.placeholder(float_type, shape=())
        self.wd_rate2_ph = tf.placeholder(float_type, shape=())
        self.lrn_rate_ph = tf.placeholder(float_type, shape=())

        if optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate_ph)
        elif optimizer == 'rmsp':
            self.optimizer = tf.train.RMSPropOptimizer(self.lrn_rate_ph)
        elif optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lrn_rate_ph, self.momentum)
        elif optimizer == 'mom':
            self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate_ph, self.momentum)
        else:
            print('unknown optimizer name: ', self.optimizer)
            print('Default to Momentum.')
            self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate_ph, self.momentum)

    def inference(self, images):
        raise NotImplementedError("Implement this method.")

    def apply_gradients_from_cost(self, cost):
        return self.optimizer.minimize(cost, self.global_step, colocate_gradients_with_ops=True)

    def build_train_ops(self, images, labels):
        raise NotImplementedError("Implement this method.")

    def _decay(self, mode):
        """L2 weight decay loss."""
        print('\n< weight decay info >\n')
        if mode == 0:
            print('Applying L2 regularization...')
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
        return tf.add(tf.multiply(self.wd_rate_ph, losses_existing_layers),
                      tf.multiply(self.wd_rate2_ph, losses_new_layers),
                      name='weight_decay')

