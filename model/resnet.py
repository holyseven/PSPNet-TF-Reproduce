from __future__ import print_function, division, absolute_import
"""ResNet model.
Related papers:
https://arxiv.org/pdf/1512.03385v1.pdf
"""

import tensorflow as tf
from model import utils, network_base


class ResNet(network_base.Network):
    """ResNet model."""

    def __init__(self, num_classes, lrn_rate_placeholder, wd_rate_placeholder, wd_rate_placeholder2,
                 mode='train', initializer='he', fix_blocks=0,
                 RV=False, fine_tune_filename=None,
                 bn_ema=0.9, bn_epsilon=1e-5, norm_only=False,
                 wd_mode=0, optimizer='mom', momentum=0.9, fisher_filename=None,
                 gpu_num=1, fisher_epsilon=0, data_format='NHWC',
                 resnet='resnet_v1_101', strides=None, filters=None, num_residual_units=None, rate=None,
                 float_type=tf.float32, separate_regularization=False):
        """ResNet constructor.
        Args:
          mode: One of 'train' and 'test'.
        """
        super(ResNet, self).__init__(num_classes, lrn_rate_placeholder, wd_rate_placeholder, wd_rate_placeholder2,
                                     mode, initializer, fix_blocks,
                                     RV, fine_tune_filename,
                                     wd_mode, optimizer, momentum, fisher_filename,
                                     gpu_num, fisher_epsilon, data_format, float_type, separate_regularization)

        # ============== for bn layers ================
        self.bn_ema = bn_ema
        self.bn_epsilon = bn_epsilon
        self.bn_use_gamma = True
        self.bn_use_beta = True
        if norm_only is True:
            self.bn_use_gamma = False
            self.bn_use_beta = False

        # ============ network structure ==============
        self.strides = [2, 2, 2, 1]
        self.filters = [256, 512, 1024, 2048]
        self.num_residual_units = [3, 4, 23, 3]
        self.rate = [1, 1, 1, 1]
        if resnet is None:
            self.strides = strides
            self.filters = filters
            self.num_residual_units = num_residual_units
            self.rate = rate
        elif resnet == 'resnet_v1_50':
            self.num_residual_units = [3, 4, 6, 3]
        elif resnet == 'resnet_v1_101':
            self.num_residual_units = [3, 4, 23, 3]
        elif resnet == 'resnet_v1_152':
            self.num_residual_units = [3, 8, 36, 3]
        else:
            raise ValueError('Unknown resnet structure: %s' % resnet)

    def inference(self, images):
        raise NotImplementedError()

        print('================== Resnet structure =======================')
        print('num_residual_units: ', self.num_residual_units)
        print('channels in each block: ', self.filters)
        print('stride in each block: ', self.strides)
        print('================== constructing network ====================')

        x = utils.input_data(images, self.data_format)
        x = tf.cast(x, self.float_type)

        with tf.variable_scope('conv1'):
            trainable_ = False if self.fix_blocks > 0 else True
            self.fix_blocks -= 1
            x = utils.conv2d_same(x, 64, 7, 2,
                                  trainable=trainable_, data_format=self.data_format, initializer=self.initializer,
                                  float_type=self.float_type)
            x = utils.batch_norm('BatchNorm', x, trainable_, self.data_format, self.mode,
                                 use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                 bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                 float_type=self.float_type)
            x = utils.relu(x)
            x = utils.max_pool(x, 3, 2, self.data_format)

        for block_index in range(len(self.num_residual_units)):
            for unit_index in range(self.num_residual_units[block_index]):
                with tf.variable_scope('block%d' % (block_index+1)):
                    with tf.variable_scope('unit_%d' % (unit_index+1)):
                        stride = 1
                        if unit_index == self.num_residual_units[block_index] - 1:
                            stride = self.strides[block_index]

                        trainable_ = False if self.fix_blocks > 0 else True
                        self.fix_blocks -= 1
                        x = utils.bottleneck_residual(x, self.filters[block_index], stride,
                                                      data_format=self.data_format, initializer=self.initializer,
                                                      rate=self.rate[block_index],
                                                      trainable=trainable_,
                                                      bn_mode=self.mode,
                                                      bn_use_gamma=self.bn_use_gamma, bn_use_beta=self.bn_use_beta,
                                                      bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                                      float_type=self.float_type)

        with tf.variable_scope('logits'):
            x = utils.global_avg_pool(x, self.data_format)
            self.logits = utils.fully_connected(x, self.num_classes,
                                                trainable=True,
                                                data_format=self.data_format,
                                                initializer=self.initializer,
                                                float_type=self.float_type)
            self.logits = tf.reshape(self.logits, (-1, self.num_classes))
            self.predictions = tf.nn.softmax(self.logits)

        return self.logits
