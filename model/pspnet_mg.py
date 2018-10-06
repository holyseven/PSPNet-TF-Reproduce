from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from . import utils_mg as utils
from . import network_base


class PSPNetMG(network_base.Network):

    def __init__(self, num_classes,
                 mode='train', initializer='he', fine_tune_filename=None,
                 wd_mode=0, optimizer='mom', momentum=0.9, gpu_num=1, data_format='NHWC',
                 resnet='resnet_v1_101', bn_mode='gather',
                 float_type=tf.float32, has_aux_loss=True,
                 train_like_in_paper=True, structure_in_paper=False,
                 new_layer_names=None, loss_type='normal', consider_dilated=False):
        """ResNet constructor.
        Args:
          mode: One of 'train' and 'test'.
        """
        super(PSPNetMG, self).__init__(num_classes,
                                       mode, initializer, fine_tune_filename,
                                       wd_mode, optimizer, momentum,
                                       gpu_num, data_format, float_type)
        # assert float_type == tf.float32
        self.loss_type = loss_type

        # ============ network structure ==============
        self.resnet = resnet
        self.filters = [256, 512, 1024, 2048]
        self.rate = [1, 1, 2, 4]
        self.strides = [1, 2, 1, 1]
        if resnet == 'resnet_v1_50':
            self.num_residual_units = [3, 4, 6, 3]
        elif resnet == 'resnet_v1_101':
            self.num_residual_units = [3, 4, 23, 3]
        elif resnet == 'resnet_v1_152':
            self.num_residual_units = [3, 8, 36, 3]
        else:
            raise ValueError('Unknown resnet structure: %s' % resnet)

        self.has_aux_loss = has_aux_loss
        if self.mode != 'train':
            self.has_aux_loss = False

        self.bn_mode = self.mode + '_' + bn_mode
        self.train_like_in_paper = train_like_in_paper
        self.structure_in_paper = structure_in_paper
        if new_layer_names is None:
            self.new_layers_names = ['block%d/unit_%d' % (len(self.filters), self.num_residual_units[-1]+1),
                                     'block%d/unit_%d' % (len(self.filters)-1, self.num_residual_units[-2] + 1),
                                     'logits', 'psp']
        else:
            self.new_layers_names = new_layer_names

        if consider_dilated:
            self.sp_group = ['block3', 'block4', 'logits', 'psp']
        else:
            self.sp_group = self.new_layers_names

    def build_train_ops(self, images, labels):
        # < outputs >
        train_ops = []

        # <construct network>
        with tf.variable_scope(self.resnet):
            logits, auxiliary_x = pspnet_with_list(images, self.num_classes, self.bn_mode, True,
                                                   self.resnet, self.structure_in_paper, self.initializer,
                                                   self.data_format, self.float_type,
                                                   return_list_activations=False, verbo=True)
        if self.loss_type == 'normal':
            print('normal cross entropy with softmax ... ')
            loss_type = self._normal_loss
        elif self.loss_type == 'focal_1':
            print('focal loss: cross entropy with softmax ... ')
            loss_type = self._focal_loss_1
        else:
            raise NotImplementedError()

        num_valide_pixel = 0
        losses = []
        for i in range(len(labels)):
            with tf.device('/gpu:%d' % i):
                # < valid pixel indice >
                label = tf.reshape(labels[i], [-1, ])
                indice = tf.where(tf.logical_and(tf.less(label, self.num_classes), tf.greater_equal(label, 0)))
                label = tf.cast(tf.gather(label, indice), tf.int32)
                num_valide_pixel += tf.shape(label)[0]

                # < aux logits >
                aux_logit = tf.reshape(auxiliary_x[i], [-1, self.num_classes])
                aux_logit = tf.gather(aux_logit, indice)

                # < logits >
                logit = tf.reshape(logits[i], [-1, self.num_classes])
                logit = tf.gather(logit, indice)
                prediction = tf.argmax(tf.nn.softmax(logit), axis=-1)
                from tensorflow.contrib.metrics import streaming_mean_iou
                self.precision_op, update_op = streaming_mean_iou(prediction, label, num_classes=self.num_classes)

                # < loss >
                loss = loss_type(logit, label)
                auxiliary_loss = 0.4 * loss_type(aux_logit, label)

                train_ops.append(update_op)
                losses.append(loss)
                losses.append(auxiliary_loss)

        num_valide_pixel = tf.cast(num_valide_pixel, tf.float32)
        self.loss = tf.truediv(tf.reduce_sum(losses), num_valide_pixel)
        self.wd = self._decay(self.wd_mode)
        total_cost = self.loss + self.wd

        if self.train_like_in_paper:
            train_ops.append(self.compute_gradients_different_lr(total_cost))
        else:
            train_ops.append(self.apply_gradients_from_cost(total_cost))

        return train_ops

    def build_eval_ops(self, images, labels):
        return

    def inference(self, images):
        # <construct network>
        with tf.variable_scope(self.resnet):
            logits = pspnet_with_list(images, self.num_classes, self.bn_mode, False,
                                      self.resnet, self.structure_in_paper, self.initializer,
                                      self.data_format, self.float_type,
                                      return_list_activations=False, verbo=True)
        return logits

    def compute_gradients_different_lr(self, cost):
        def get_different_variables(new_layers_names):
            existing_weights = []
            new_normal_weights = []
            new_bias_weights = []
            for v in tf.trainable_variables():
                if any(elem in v.name for elem in new_layers_names):
                    if 'bias' in v.name:
                        new_bias_weights.append(v)
                    else:
                        new_normal_weights.append(v)
                    continue
                existing_weights.append(v)

            return existing_weights, new_normal_weights, new_bias_weights

        if self.optimizer == 'sgd':
            opt_existing = tf.train.GradientDescentOptimizer(self.lrn_rate_ph)
            opt_new_norm = tf.train.GradientDescentOptimizer(self.lrn_rate_ph * 10)
            opt_new_bias = tf.train.GradientDescentOptimizer(self.lrn_rate_ph * 20)
        elif self.optimizer == 'mom':
            opt_existing = tf.train.MomentumOptimizer(self.lrn_rate_ph, self.momentum)
            opt_new_norm = tf.train.MomentumOptimizer(self.lrn_rate_ph * 10, self.momentum)
            opt_new_bias = tf.train.MomentumOptimizer(self.lrn_rate_ph * 20, self.momentum)
        elif self.optimizer == 'rmsp':
            opt_existing = tf.train.RMSPropOptimizer(self.lrn_rate_ph)
            opt_new_norm = tf.train.RMSPropOptimizer(self.lrn_rate_ph)
            opt_new_bias = tf.train.RMSPropOptimizer(self.lrn_rate_ph)
        elif self.optimizer == 'adam':
            opt_existing = tf.train.AdamOptimizer(self.lrn_rate_ph, self.momentum)
            opt_new_norm = tf.train.AdamOptimizer(self.lrn_rate_ph, self.momentum)
            opt_new_bias = tf.train.AdamOptimizer(self.lrn_rate_ph, self.momentum)
        else:
            opt_existing = tf.train.MomentumOptimizer(self.lrn_rate_ph, self.momentum)
            opt_new_norm = tf.train.MomentumOptimizer(self.lrn_rate_ph * 10, self.momentum)
            opt_new_bias = tf.train.MomentumOptimizer(self.lrn_rate_ph * 20, self.momentum)

        existing_weights, new_normal_weights, new_bias_weights = get_different_variables(self.new_layers_names)

        grads = tf.gradients(cost, existing_weights + new_normal_weights + new_bias_weights,
                             colocate_gradients_with_ops=True)

        grads_existing = grads[:len(existing_weights)]
        grads_new_norm = grads[len(existing_weights): (len(existing_weights) + len(new_normal_weights))]
        grads_new_bias = grads[(len(existing_weights) + len(new_normal_weights)):]
        train_existing = opt_existing.apply_gradients(zip(grads_existing, existing_weights))
        train_new_norm = opt_new_norm.apply_gradients(zip(grads_new_norm, new_normal_weights))
        train_new_bias = opt_new_bias.apply_gradients(zip(grads_new_bias, new_bias_weights))

        apply_op = tf.group(train_existing, train_new_norm, train_new_bias)
        return apply_op

    def _normal_loss(self, logits, labels):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_sum(xent)

    def _focal_loss_1(self, logits, labels, gamma=2):
        proba = tf.nn.softmax(logits, dim=-1)
        if gamma == 2:
            proba_gamma = tf.square(1 - proba)
        else:
            proba_gamma = tf.pow(1-proba, gamma)
        _t = tf.multiply(proba_gamma, tf.one_hot(labels, self.num_classes))
        _t = - 1.0 * tf.multiply(_t, tf.log(proba))
        return tf.reduce_sum(_t)


def pspnet_with_list(images, output_num_classes, bn_mode, has_aux_loss,
                     resnet='resnet_v1_101', three_convs_beginning=False, initializer='he',
                     data_format='NHWC', float_type=tf.float32,
                     return_list_activations=False, verbo=True):
    if not verbo:
        print('\n< The network will be constructed quietly in', bn_mode, 'mode. >\n')

    list_activations = []
    image_shape = images[0].get_shape().as_list()
    image_size = image_shape[1]
    output_size = tf.cast(tf.stack([image_size // 8, image_size // 8]), tf.int32)
    pool_rates = [6, 3, 2, 1]
    pool_size = image_size // 8 // np.array(pool_rates)
    if resnet == 'resnet_v1_101':
        num_residual_units = [3, 4, 23, 3]
        rate = [1, 1, 2, 4]
        strides = [1, 2, 1, 1]
        filters = [256, 512, 1024, 2048]
    elif resnet == 'resnet_v1_50':
        num_residual_units = [3, 4, 6, 3]
        rate = [1, 1, 2, 4]
        strides = [1, 2, 1, 1]
        filters = [256, 512, 1024, 2048]
    else:
        raise NotImplementedError('Does not support other structures than resnet_v1_101 or resnet_v1_50.')

    if verbo:
        print('\n< Resnet structure >\n')
        print('num_residual_units: ', num_residual_units)
        print('rates in each atrous convolution: ', rate)
        print('stride in each block: ', strides)
        print('channels in each block: ', filters)

    x = utils.input_data(images, data_format)
    # < first block >
    if three_convs_beginning:
        with tf.variable_scope('conv1_1'):
            x = utils.conv2d_same(x, 64, 3, 2, data_format=data_format, initializer=initializer,
                                  float_type=float_type)
            x = utils.batch_norm(x, bn_mode, data_format, float_type)
            x = utils.relu(x)
            if verbo: print('shape after conv1_1: ', x[0].get_shape())

        with tf.variable_scope('conv1_2'):
            x = utils.conv2d_same(x, 64, 3, 1, data_format=data_format, initializer=initializer,
                                  float_type=float_type)
            x = utils.batch_norm(x, bn_mode, data_format, float_type)
            x = utils.relu(x)
            if verbo: print('shape after conv1_2: ', x[0].get_shape())

        with tf.variable_scope('conv1_3'):
            x = utils.conv2d_same(x, 128, 3, 1, data_format=data_format, initializer=initializer,
                                  float_type=float_type)
            x = utils.batch_norm(x, bn_mode, data_format, float_type)
            x = utils.relu(x)
            if verbo: print('shape after conv1_3: ', x[0].get_shape())
    else:
        with tf.variable_scope('conv1'):
            x = utils.conv2d_same(x, 64, 7, 2, data_format=data_format, initializer=initializer,
                                  float_type=float_type)
            x = utils.batch_norm(x, bn_mode, data_format, float_type)
            x = utils.relu(x)

    list_activations.append(x)
    x = utils.max_pool(x, 3, 2, data_format)
    if verbo: print('shape after pool1: ', x[0].get_shape())

    for block_index in range(len(num_residual_units)):
        for unit_index in range(num_residual_units[block_index]):
            with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 1)):
                stride = 1
                # this is the original version of resnet and it can save more spatial information
                if unit_index == 0:
                    stride = strides[block_index]

                x = utils.bottleneck_residual(x, filters[block_index], stride,
                                              float_type=float_type, data_format=data_format,
                                              initializer=initializer,
                                              rate=rate[block_index],
                                              bn_mode=bn_mode)

            # auxiliary loss operations, after block3/unit23, add block3/unit24 and auxiliary loss.
            if block_index + 1 == 3 and unit_index + 1 == num_residual_units[-2] and has_aux_loss:
                with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 2)):
                    # new layers
                    auxiliary_x = utils.conv2d_same(x, 256, 3, 1, trainable=True, data_format=data_format,
                                                    initializer=initializer, float_type=float_type)
                    auxiliary_x = utils.batch_norm(auxiliary_x, bn_mode,
                                                   data_format=data_format, float_type=float_type)
                    auxiliary_x = utils.relu(auxiliary_x)
                    auxiliary_x = utils.dropout(auxiliary_x, keep_prob=0.9)
                with tf.variable_scope('aux_logits'):
                    # new layers
                    if verbo: print('aux_logits: ', auxiliary_x[0].get_shape())
                    auxiliary_x = utils.fully_connected(auxiliary_x, output_num_classes, trainable=True,
                                                        data_format=data_format, initializer=initializer,
                                                        float_type=float_type)

                with tf.variable_scope('aux_up_sample'):
                    auxiliary_x = utils.resize_images(auxiliary_x,
                                                      image_shape[1:3] if data_format == 'NHWC'
                                                      else image_shape[2:4],
                                                      data_format)
                    if verbo: print('upsampled auxiliary_x for loss function: ', auxiliary_x[0].get_shape())

            # psp operations, after block4/unit3, add psp/pool6321/ and block4/unit4.
            if block_index + 1 == 4 and unit_index + 1 == num_residual_units[-1]:
                # 4 pooling layers.
                to_concat = [x]
                with tf.variable_scope('psp'):
                    for pool_index in range(len(pool_size)):
                        with tf.variable_scope('pool%d' % pool_rates[pool_index]):
                            pool_output = utils.avg_pool(x,
                                                         utils.stride_arr(pool_size[pool_index], data_format),
                                                         utils.stride_arr(pool_size[pool_index], data_format),
                                                         'SAME', data_format=data_format)
                            pool_output = utils.conv2d_same(pool_output, filters[block_index] // 4, 1, 1,
                                                            trainable=True,
                                                            data_format=data_format,
                                                            initializer=initializer,
                                                            float_type=float_type)
                            pool_output = utils.batch_norm(pool_output, bn_mode,
                                                           data_format=data_format, float_type=float_type)
                            if verbo:
                                print('pool%d' % pool_rates[pool_index], 'pooled size: ', pool_output[0].get_shape())
                            pool_output = utils.relu(pool_output)
                            pool_output = utils.resize_images(pool_output, output_size, data_format)
                            if verbo:
                                print('pool%d' % pool_rates[pool_index], 'output size: ', pool_output[0].get_shape())
                            to_concat.append(pool_output)
                x = utils.concat(to_concat, axis=1 if data_format == 'NCHW' else 3)

                # block4/unit4
                with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 2)):
                    # new layers
                    x = utils.conv2d_same(x, filters[block_index] // 4, 3, 1, trainable=True,
                                          data_format=data_format, initializer=initializer,
                                          float_type=float_type)
                    x = utils.batch_norm(x, bn_mode, data_format=data_format, float_type=float_type)
                    x = utils.relu(x)
                    if 'train' in bn_mode:
                        x = utils.dropout(x, keep_prob=0.9)

            list_activations.append(x)

        if verbo: print('shape after block %d: ' % (block_index + 1), x[0].get_shape())

    with tf.variable_scope('logits'):
        # new layers
        if verbo: print('logits: ', x[0].get_shape(), )
        logits = utils.fully_connected(x, output_num_classes, trainable=True,
                                       data_format=data_format, initializer=initializer,
                                       float_type=float_type)
        list_activations.append(logits)

    with tf.variable_scope('up_sample'):
        logits = utils.resize_images(logits, image_shape[1:3] if data_format == 'NHWC' else image_shape[2:4],
                                     data_format)
        if verbo: print('logits after upsampling: ', logits[0].get_shape())
        list_activations.append(logits)

    if return_list_activations:
        if has_aux_loss:
            return logits, auxiliary_x, list_activations
        else:
            return logits, list_activations
    else:
        if has_aux_loss:
            return logits, auxiliary_x
        else:
            return logits