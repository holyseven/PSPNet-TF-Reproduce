import numpy as np
import tensorflow as tf
from . import utils_mg as utils
from database import helper
from tensorflow.contrib.metrics import streaming_mean_iou


class PSPNetMG(object):
    def __init__(self, num_classes, resnet='resnet_v1_101', gpu_num=1,
                 initializer='he',
                 wd_mode=0, fine_tune_filename=None,
                 optimizer='mom', momentum=0.9,
                 train_like_in_caffe=False, three_convs_beginning=False,
                 new_layer_names=None, loss_type='normal', consider_dilated=False):

        # < initialize variables in advance>
        self.global_step = None
        self.logits = None
        self.num_classes = num_classes
        self.momentum = momentum
        self.gpu_num = gpu_num
        self.initializer = initializer
        self.loss_type = loss_type

        # < initialize some placeholders >
        self.wd_rate_ph = tf.placeholder(utils.float_type, shape=())
        self.wd_rate2_ph = tf.placeholder(utils.float_type, shape=())
        self.lrn_rate_ph = tf.placeholder(utils.float_type, shape=())
        #
        if optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate_ph)
        elif optimizer == 'rmsp':
            self.optimizer = tf.train.RMSPropOptimizer(self.lrn_rate_ph)
        elif optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lrn_rate_ph, self.momentum)
        elif optimizer == 'mom':
            self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate_ph, self.momentum)
        else:
            print('\t[verbo] unknown optimizer name: ', self.optimizer)
            print('\t[verbo] Default to Momentum.')
            self.optimizer = tf.train.MomentumOptimizer(self.lrn_rate_ph, self.momentum)

        # < ============ network structure ============== >
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

        print('[info] < Resnet structure hyperparameters>')
        print('\t - num_residual_units: ', self.num_residual_units)
        print('\t - channels in each block: ', self.filters)
        print('\t - strides in each block: ', self.strides)
        print('\t - rates in each block: ', self.rate)

        # < for L2-SP weight regularization>
        if new_layer_names is None:
            self.new_layers_names = ['block%d/unit_%d' % (len(self.filters), self.num_residual_units[-1]+1),
                                     'block%d/unit_%d' % (len(self.filters)-1, self.num_residual_units[-2] + 1),
                                     'logits', 'psp']
        else:
            splits = new_layer_names.split(',')
            if len(splits) == 1 and splits[0] == '':
                self.new_layers_names = []
            else:
                self.new_layers_names = splits

        if consider_dilated:
            self.sp_group = ['block3', 'block4', 'logits', 'psp']
        else:
            self.sp_group = self.new_layers_names
        self.sp_group = self.new_layers_names
        self.fine_tune_filename = fine_tune_filename
        self.wd_mode = wd_mode

        self.train_like_in_caffe = train_like_in_caffe
        self.three_convs_beginning = three_convs_beginning

        self.reuse = False

    def build_train_ops(self, images, labels, weights=None):
        train_ops = []

        # <construct network>
        with tf.variable_scope(self.resnet, reuse=self.reuse):
            logits, aux_logits = pspnet_with_list(images, self.num_classes,
                                                  training=True,
                                                  resnet=self.resnet,
                                                  three_convs_beginning=self.three_convs_beginning,
                                                  initializer=self.initializer,
                                                  return_list_activations=False, verbo=True)
        self.reuse = True

        # < losses >
        losses = []
        aux_losses = []
        num_valide_pixel = 0
        for i in range(len(labels)):
            with tf.device('/gpu:%d' % i):
                # < valid pixel indice >
                label = tf.reshape(labels[i], [-1, ])
                indice = tf.where(tf.logical_and(tf.less(label, self.num_classes), tf.greater_equal(label, 0)))
                label = tf.cast(tf.gather(label, indice), tf.int32)
                num_valide_pixel += tf.shape(label)[0]
                if weights is not None:
                    weights_i = tf.reshape(weights[i], [-1, ])
                    weights_i = tf.gather(weights_i, indice)
                else:
                    weights_i = None

                # < aux logits >
                aux_logit = tf.reshape(aux_logits[i], [-1, self.num_classes])
                aux_logit = tf.gather(aux_logit, indice)

                # < logits >
                logit = tf.reshape(logits[i], [-1, self.num_classes])
                logit = tf.gather(logit, indice)
                prediction = tf.argmax(tf.nn.softmax(logit), axis=-1)
                precision_op, update_op = streaming_mean_iou(prediction, label, num_classes=self.num_classes)

                # < loss >
                loss = self._normal_loss(logit, label, weights_i)
                auxiliary_loss = 0.4 * self._normal_loss(aux_logit, label, weights_i)

                train_ops.append(update_op)
                losses.append(loss)
                aux_losses.append(auxiliary_loss)

        num_valide_pixel = tf.cast(num_valide_pixel, tf.float32)
        loss = tf.truediv(tf.reduce_sum(losses), num_valide_pixel, name='loss')
        aux_loss = tf.truediv(tf.reduce_sum(aux_losses), num_valide_pixel, name='aux_loss')
        wd = self._decay(self.wd_mode)

        total_loss = loss+aux_loss+wd
        if self.train_like_in_caffe:
            train_ops.append(self._compute_gradients_different_lr(total_loss))
        else:
            train_ops.append(self._apply_gradients_from_cost(total_loss))

        return train_ops, [loss, aux_loss, wd], [precision_op]

    def build_forward_ops(self, list_input):
        assert type(list_input) == list

        # <construct network>
        with tf.variable_scope(self.resnet, reuse=self.reuse):
            logits = pspnet_with_list(list_input, self.num_classes,
                                      training=False,
                                      resnet=self.resnet,
                                      three_convs_beginning=self.three_convs_beginning,
                                      initializer=self.initializer,
                                      return_list_activations=False, verbo=False)
            probas = utils.softmax(logits)

        self.reuse = True

        return probas

    def build_inference_ops(self, one_image, crop_size):
        one_image = tf.convert_to_tensor(one_image)
        shape_image = tf.shape(one_image)
        one_image_3D = tf.reshape(one_image, [shape_image[-3], shape_image[-2], shape_image[-1]])
        # shape of one_image: [H, W, 3]; if it cannot reshape, there is something wrong.

        H, W, channel = (1024, 2048, 3)  # only for Cityscapes original images.

        # < split >
        crop_heights = helper.decide_intersection(H, crop_size)
        crop_widths = helper.decide_intersection(W, crop_size)
        output_list = []
        for i, height in enumerate(crop_heights):
            for j, width in enumerate(crop_widths):
                image_crop = one_image_3D[height:height + crop_size, width:width + crop_size]
                image_crop.set_shape((crop_size, crop_size, 3))
                output_list.append(image_crop)
        image_crops = tf.stack(output_list)

        # <construct network>
        with tf.variable_scope(self.resnet, reuse=self.reuse):
            logits = pspnet_with_list([image_crops], self.num_classes,
                                      training=False,
                                      resnet=self.resnet,
                                      three_convs_beginning=self.three_convs_beginning,
                                      initializer=self.initializer,
                                      return_list_activations=False, verbo=False)
            probas = tf.nn.softmax(logits)[0]
            # shape of probas: (N, crop_size, crop_size, num_classes)

        self.reuse = True

        # < reassemble >
        reassemble_proba = tf.zeros(shape=(H, W, self.num_classes))
        for i, height in enumerate(crop_heights):
            for j, width in enumerate(crop_widths):
                # use padding then add.
                reassemble_proba += tf.image.pad_to_bounding_box(probas[i*len(crop_widths)+j], height, width, H, W)
                # reassemble_proba[height:height+crop_size, width:width+crop_size] += probas[i*len(crop_widths)+j]

        return reassemble_proba

    def build_eval_ops(self, images, labels):
        return

    def _normal_loss(self, logits, labels, weights=None):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        if weights is not None:
            xent = tf.multiply(xent, weights)
        return tf.reduce_sum(xent)

    def _apply_gradients_from_cost(self, cost):
        return self.optimizer.minimize(cost, self.global_step, colocate_gradients_with_ops=True)

    def _compute_gradients_different_lr(self, cost):
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

    def _decay(self, mode):
        """L2 weight decay loss."""
        print('\n\t[verbo] < weight decay info >')
        if mode == 0:
            print('\t - Applying L2 regularization...')
            losses_existing_layers = 0.0
            losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.sp_group):
                        losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    losses_existing_layers += tf.nn.l2_loss(v)
        elif mode == 1:
            print('\t[verbo] applying L2-SP regularization... with exception of', self.sp_group)
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            losses_existing_layers = 0.0
            losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.sp_group):
                        losses_new_layers += tf.nn.l2_loss(v)
                        print('\t[verbo] layers with L2:', v.name)
                        continue

                    name = v.name.split(':')[0]
                    if reader.has_tensor(name):
                        pre_trained_weights = reader.get_tensor(name)
                        print('\t[verbo] layers with L2-SP:', v.name)
                    else:
                        raise KeyError('not find %s' % name)

                    losses_existing_layers += tf.nn.l2_loss(v - pre_trained_weights)
        elif self.wd_mode == 2:
            print('\t[verbo] applying L2-SP considering the normalization... with exception of', self.sp_group)
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            losses_existing_layers = 0.0
            losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.sp_group):
                        losses_new_layers += tf.nn.l2_loss(v)
                        print('\t[verbo] layers with L2 :', v.name)
                        continue

                    name = v.name.split(':')[0]
                    if reader.has_tensor(name):
                        pre_trained_weights = reader.get_tensor(name)
                        print('\t[verbo] layers with L2-SP-norm:', v.name)
                    else:
                        raise KeyError('not find %s' % name)

                    norm = tf.reduce_sum(tf.multiply(pre_trained_weights, pre_trained_weights),
                                         axis=[0, 1, 2], keepdims=True)
                    cos = tf.reduce_sum(tf.multiply(v, pre_trained_weights), axis=[0, 1, 2], keepdims=True)
                    projection_v = tf.truediv(tf.multiply(cos, pre_trained_weights), norm)
                    perp = v - projection_v

                    v_shape = v.get_shape().as_list()
                    alpha_k = float(np.sqrt(v_shape[0] * v_shape[1] * v_shape[2]))

                    losses_existing_layers += tf.nn.l2_loss(perp) * alpha_k
        elif self.wd_mode == 3:
            print('\t[verbo] applying L2-SP considering the normalization... with exception of', self.sp_group)
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            losses_existing_layers = 0.0
            losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.sp_group):
                        losses_new_layers += tf.nn.l2_loss(v)
                        print('\t[verbo] layers with L2 :', v.name)
                        continue

                    name = v.name.split(':')[0]
                    if reader.has_tensor(name):
                        pre_trained_weights = reader.get_tensor(name)
                        print('\t[verbo] layers with L2-SP-norm:', v.name)
                    else:
                        raise KeyError('not find %s' % name)

                    # equivalent to this:
                    cos = tf.reduce_sum(tf.multiply(v, pre_trained_weights), axis=[0, 1, 2])
                    cos_square = tf.square(cos)
                    norm = tf.reduce_sum(tf.multiply(pre_trained_weights, pre_trained_weights), axis=[0, 1, 2])
                    losses_existing_layers += tf.nn.l2_loss(v) - 0.5 * tf.reduce_sum(tf.truediv(cos_square, norm))

                    # norm = tf.reduce_sum(tf.multiply(pre_trained_weights, pre_trained_weights),
                    #                      axis=[0, 1, 2], keepdims=True)
                    # cos = tf.reduce_sum(tf.multiply(v, pre_trained_weights), axis=[0, 1, 2], keepdims=True)
                    # projection_v = tf.truediv(tf.multiply(cos, pre_trained_weights), norm)
                    # perp = v - projection_v
                    # losses_existing_layers += tf.nn.l2_loss(perp)
        else:
            print('\t[verbo] No regularization...')
            return tf.convert_to_tensor(0.0)
        return tf.add(tf.multiply(self.wd_rate_ph, losses_existing_layers),
                      tf.multiply(self.wd_rate2_ph, losses_new_layers),
                      name='weight_decay')


def pspnet_with_list(images, output_num_classes, training=True,
                     resnet='resnet_v1_101', three_convs_beginning=False, initializer='he',
                     return_list_activations=False, verbo=True):
    if training:
        bn_stat_mode = 'gather'
        has_aux_loss = True
    else:
        bn_stat_mode = 'frozen'
        has_aux_loss = False

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

    if not verbo:
        print('\n[info] The network will be constructed quietly.')
    else:
        print('[info] The network will be constructed with verbose.')

    x = utils.input_data(images)
    x = utils.cast(x)

    # < first block >
    if three_convs_beginning:
        with tf.variable_scope('conv1_1'):
            x = utils.conv2d_same(x, 64, 3, 2, initializer=initializer)
            x = utils.batch_norm(x, bn_stat_mode)
            x = utils.relu(x)
            if verbo: print('\t[verbo] shape after conv1_1: ', x[0].get_shape())

        with tf.variable_scope('conv1_2'):
            x = utils.conv2d_same(x, 64, 3, 1, initializer=initializer)
            x = utils.batch_norm(x, bn_stat_mode)
            x = utils.relu(x)
            if verbo: print('\t[verbo] shape after conv1_2: ', x[0].get_shape())

        with tf.variable_scope('conv1_3'):
            x = utils.conv2d_same(x, 128, 3, 1, initializer=initializer)
            x = utils.batch_norm(x, bn_stat_mode)
            x = utils.relu(x)
            if verbo: print('\t[verbo] shape after conv1_3: ', x[0].get_shape())
    else:
        with tf.variable_scope('conv1'):
            x = utils.conv2d_same(x, 64, 7, 2, initializer=initializer)
            x = utils.batch_norm(x, bn_stat_mode)
            x = utils.relu(x)

    list_activations.append(x)
    x = utils.max_pool(x, 3, 2)
    if verbo: print('\t[verbo] shape after pool1: ', x[0].get_shape())

    for block_index in range(len(num_residual_units)):
        for unit_index in range(num_residual_units[block_index]):
            with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 1)):
                stride = 1
                # this is the original version of resnet and it can save more spatial information
                if unit_index == 0:
                    stride = strides[block_index]

                x = utils.bottleneck_residual(x, filters[block_index], stride,
                                              rate=rate[block_index],
                                              initializer=initializer,
                                              bn_stat_mode=bn_stat_mode)

            # auxiliary loss operations, after block3/unit23, add block3/unit24 and auxiliary loss.
            if block_index + 1 == 3 and unit_index + 1 == num_residual_units[-2] and has_aux_loss:
                with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 2)):
                    # new layers
                    auxiliary_x = utils.conv2d_same(x, 256, 3, 1, initializer=initializer)
                    auxiliary_x = utils.batch_norm(auxiliary_x, bn_stat_mode)
                    auxiliary_x = utils.relu(auxiliary_x)
                    auxiliary_x = utils.dropout(auxiliary_x, keep_prob=0.9)
                with tf.variable_scope('aux_logits'):
                    # new layers
                    if verbo: print('\t[verbo] aux_logits: ', auxiliary_x[0].get_shape())
                    auxiliary_x = utils.fully_connected(auxiliary_x, output_num_classes, initializer=initializer)

                with tf.variable_scope('aux_up_sample'):
                    auxiliary_x = utils.resize_images(auxiliary_x,
                                                      image_shape[1:3])
                    if verbo: print('\t[verbo] upsampled auxiliary_x for loss function: ', auxiliary_x[0].get_shape())

            # psp operations, after block4/unit3, add psp/pool6321/ and block4/unit4.
            if block_index + 1 == 4 and unit_index + 1 == num_residual_units[-1]:
                # 4 pooling layers.
                to_concat = [x]
                with tf.variable_scope('psp'):
                    for pool_index in range(len(pool_size)):
                        with tf.variable_scope('pool%d' % pool_rates[pool_index]):
                            pool_output = utils.avg_pool(x,
                                                         utils.stride_arr(pool_size[pool_index]),
                                                         utils.stride_arr(pool_size[pool_index]),
                                                         'SAME')
                            if verbo:
                                print('\t[verbo] pool%d' % pool_rates[pool_index], 'pooled size: ',
                                      pool_output[0].get_shape())
                            pool_output = utils.conv2d_same(pool_output, filters[block_index] // 4, 1, 1,
                                                            initializer=initializer)
                            pool_output = utils.batch_norm(pool_output, bn_stat_mode)
                            pool_output = utils.relu(pool_output)
                            pool_output = utils.resize_images(pool_output, output_size)
                            if verbo:
                                print('\t[verbo] pool%d' % pool_rates[pool_index], 'output size: ',
                                      pool_output[0].get_shape())
                            to_concat.append(pool_output)
                x = utils.concat(to_concat, axis=3)

                # block4/unit4
                with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 2)):
                    # new layers
                    x = utils.conv2d_same(x, filters[block_index] // 4, 3, 1, initializer=initializer)
                    x = utils.batch_norm(x, bn_stat_mode)
                    x = utils.relu(x)
                    if training:
                        x = utils.dropout(x, keep_prob=0.9)

            list_activations.append(x)

        if verbo: print('\t[verbo] shape after block %d: ' % (block_index + 1), x[0].get_shape())

    with tf.variable_scope('logits'):
        # new layers
        if verbo: print('\t[verbo] logits: ', x[0].get_shape(), )
        logits = utils.fully_connected(x, output_num_classes, initializer=initializer)
        list_activations.append(logits)

    with tf.variable_scope('up_sample'):
        logits = utils.resize_images(logits, image_shape[1:3])
        if verbo: print('\t[verbo] logits after upsampling: ', logits[0].get_shape())
        list_activations.append(logits)

    if not verbo:
        print('[info] Done quietly.\n')

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

