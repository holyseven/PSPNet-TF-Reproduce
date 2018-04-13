import numpy as np
import tensorflow as tf
import utils_mg as utils
import resnet


class PSPNetMG(resnet.ResNet):

    def __init__(self, num_classes, lrn_rate_placeholder, wd_rate_placeholder, wd_rate_placeholder2,
                 mode='train', initializer='he', fix_blocks=0,
                 RV=False, fine_tune_filename=None,
                 bn_ema=0.9, bn_epsilon=1e-5, bn_frozen=False,
                 wd_mode=0, optimizer='mom', momentum=0.9, fisher_filename=None,
                 norm_only=False, gpu_num=1, fisher_epsilon=0, data_format='NHWC',
                 resnet='resnet_v1_101', strides=None, filters=None, num_residual_units=None, rate=None,
                 float_type=tf.float32, has_aux_loss=True, train_like_in_paper=True, structure_in_paper=False,
                 resize_images_method='bilinear', new_layer_names=None, loss_type='normal'):
        """ResNet constructor.
        Args:
          mode: One of 'train' and 'test'.
        """
        super(PSPNetMG, self).__init__(num_classes, lrn_rate_placeholder, wd_rate_placeholder, wd_rate_placeholder2,
                                       mode, initializer, fix_blocks,
                                       RV, fine_tune_filename,
                                       bn_ema, bn_epsilon, norm_only,
                                       wd_mode, optimizer, momentum, fisher_filename,
                                       gpu_num, fisher_epsilon, data_format,
                                       resnet, strides, filters, num_residual_units, rate,
                                       float_type)
        assert bn_frozen == 0
        assert float_type == tf.float32
        self.loss_type = loss_type
        self.bn_frozen = bn_frozen
        # ============ network structure ==============
        self.rate = [1, 1, 2, 4]
        self.strides = [1, 2, 1, 1]
        if resnet is None:
            print '... ERROR ... PSPNet is based on ResNet.'

        self.has_aux_loss = has_aux_loss
        if mode != 'train':
            self.has_aux_loss = False

        self.train_like_in_paper = train_like_in_paper
        self.structure_in_paper = structure_in_paper
        self.resize_images_method = resize_images_method
        if new_layer_names is None:
            self.new_layers_names = ['block%d/unit_%d' % (len(self.filters), self.num_residual_units[-1]+1),
                                     'block%d/unit_%d' % (len(self.filters)-1, self.num_residual_units[-2] + 1),
                                     'logits', 'psp']
        else:
            self.new_layers_names = new_layer_names

    def inference(self, images):
        print '================== Resnet structure ======================='
        print 'new layer names: ', self.new_layers_names
        print 'num_residual_units: ', self.num_residual_units
        print 'channels in each block: ', self.filters
        print 'stride in each block: ', self.strides
        print 'rates in each atrous convolution: ', self.rate
        print '================== constructing network ===================='

        self.image_shape = images[0].get_shape().as_list()
        self.image_shape_tensor = tf.shape(images[0])

        x = utils.input_data(images, self.data_format)

        # TODO: height == width ?
        height = self.image_shape[1]
        image_shape = tf.cast(tf.stack([height, height]), tf.int32)
        pooling_output_size = tf.cast(tf.stack([height / 8, height / 8]), tf.int32)
        print 'size of network\'s output: ', [height, height]
        pool_rates = [6, 3, 2, 1]
        pool_size = height / 8 / np.array(pool_rates)

        bn_mode = self.mode
        if self.bn_frozen:
            bn_mode = 'test'

        print 'shape input: ', x[0].get_shape()
        if self.structure_in_paper:
            with tf.variable_scope('conv1_1'):
                trainable_ = False if self.fix_blocks > 0 else True
                self.fix_blocks -= 1

                x = utils.conv2d_same(x, 64, 3, 2,
                                      trainable=trainable_, data_format=self.data_format, initializer=self.initializer,
                                      float_type=self.float_type)
                x = utils.batch_norm('BatchNorm', x, trainable_, self.data_format, bn_mode,
                                     use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                     bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema, float_type=self.float_type)
                x = utils.relu(x)
                print 'shape after conv1_1: ', x[0].get_shape()

            with tf.variable_scope('conv1_2'):
                trainable_ = False if self.fix_blocks > 0 else True
                self.fix_blocks -= 1

                x = utils.conv2d_same(x, 64, 3, 1,
                                      trainable=trainable_, data_format=self.data_format, initializer=self.initializer,
                                      float_type=self.float_type)
                x = utils.batch_norm('BatchNorm', x, trainable_, self.data_format, bn_mode,
                                     use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                     bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema, float_type=self.float_type)
                x = utils.relu(x)
                print 'shape after conv1_2: ', x[0].get_shape()

            with tf.variable_scope('conv1_3'):
                trainable_ = False if self.fix_blocks > 0 else True
                self.fix_blocks -= 1

                x = utils.conv2d_same(x, 128, 3, 1,
                                      trainable=trainable_, data_format=self.data_format, initializer=self.initializer,
                                      float_type=self.float_type)
                x = utils.batch_norm('BatchNorm', x, trainable_, self.data_format, bn_mode,
                                     use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                     bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema, float_type=self.float_type)
                x = utils.relu(x)
                print 'shape after conv1_3: ', x[0].get_shape()

                x = utils.max_pool(x, 3, 2, self.data_format)
                print 'shape after pool1: ', x[0].get_shape()
        else:
            with tf.variable_scope('conv1'):
                trainable_ = False if self.fix_blocks > 0 else True
                self.fix_blocks -= 1
                x = utils.conv2d_same(x, 64, 7, 2,
                                      trainable=trainable_, data_format=self.data_format, initializer=self.initializer,
                                      float_type=self.float_type)
                x = utils.batch_norm('BatchNorm', x, trainable_, self.data_format, bn_mode,
                                     use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                     bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema, float_type=self.float_type)
                x = utils.relu(x)
                x = utils.max_pool(x, 3, 2, self.data_format)
            print 'shape after pool1: ', x[0].get_shape()

        for block_index in range(len(self.num_residual_units)):
            for unit_index in range(self.num_residual_units[block_index]):
                with tf.variable_scope('block%d/unit_%d' % (block_index+1, unit_index+1)):
                    stride = 1
                    # this is the original version of resnet and it can save more spatial information
                    if unit_index == 0:
                        stride = self.strides[block_index]

                    trainable_ = False if self.fix_blocks > 0 else True
                    self.fix_blocks -= 1
                    x = utils.bottleneck_residual(x, self.filters[block_index], stride,
                                                  data_format=self.data_format, initializer=self.initializer,
                                                  rate=self.rate[block_index],
                                                  trainable=trainable_,
                                                  bn_mode=bn_mode,
                                                  bn_use_gamma=self.bn_use_gamma, bn_use_beta=self.bn_use_beta,
                                                  bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                                  float_type=self.float_type)

                # auxiliary loss operations, after block3/unit23, add block3/unit24 and auxiliary loss.
                if block_index + 1 == 3 and unit_index + 1 == 23 and self.has_aux_loss:
                    with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 2)):
                        # new layers
                        auxiliary_x = utils.conv2d_same(x, 256, 3, 1, trainable=True, data_format=self.data_format,
                                                        initializer=self.initializer, float_type=self.float_type)
                        auxiliary_x = utils.batch_norm('BatchNorm', auxiliary_x,
                                                       trainable=True, data_format=self.data_format, mode=self.mode,
                                                       use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                                       bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                                       float_type=self.float_type)
                        auxiliary_x = utils.relu(auxiliary_x)
                        auxiliary_x = utils.dropout(auxiliary_x, keep_prob=0.9)
                    with tf.variable_scope('aux_logits'):
                        # new layers
                        print 'aux_logits: ', auxiliary_x[0].get_shape(),
                        auxiliary_x = utils.fully_connected(auxiliary_x, self.num_classes, trainable=True,
                                                            data_format=self.data_format, initializer=self.initializer,
                                                            float_type=self.float_type)

                        self.auxiliary_x = utils.resize_images(auxiliary_x, image_shape, self.data_format)
                        print ' auxiliary_x for loss function: ', self.auxiliary_x[0].get_shape()

                # psp operations, after block4/unit3, add psp/pool6321/ and block4/unit4.
                if block_index + 1 == 4 and unit_index + 1 == 3:
                    # 4 pooling layers.
                    to_concat = [x]
                    with tf.variable_scope('psp'):
                        for pool_index in range(len(pool_size)):
                            with tf.variable_scope('pool%d'%pool_rates[pool_index]):
                                pool_output = utils.avg_pool(x,
                                                             utils.stride_arr(pool_size[pool_index], self.data_format),
                                                             utils.stride_arr(pool_size[pool_index], self.data_format),
                                                             'SAME', data_format=self.data_format)
                                pool_output = utils.conv2d_same(pool_output, self.filters[block_index] / 4, 1, 1,
                                                                trainable=True,
                                                                data_format=self.data_format,
                                                                initializer=self.initializer,
                                                                float_type=self.float_type)
                                pool_output = utils.batch_norm('BatchNorm', pool_output,
                                                               trainable=True, data_format=self.data_format,
                                                               mode=self.mode,
                                                               use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                                               bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                                               float_type=self.float_type)
                                print 'pool%d' % pool_rates[pool_index], 'pooled size: ', pool_output[0].get_shape()
                                pool_output = utils.relu(pool_output)
                                pool_output = utils.resize_images(pool_output, pooling_output_size, self.data_format,
                                                                  self.resize_images_method)
                                print 'pool%d' % pool_rates[pool_index], 'output size: ', pool_output[0].get_shape()
                                to_concat.append(pool_output)
                    x = utils.concat(to_concat, axis=1 if self.data_format == 'NCHW' else 3)

                    # block4/unit4
                    with tf.variable_scope('block%d/unit_%d' % (block_index + 1, unit_index + 2)):
                        # new layers
                        x = utils.conv2d_same(x, self.filters[block_index] / 4, 3, 1, trainable=True,
                                              data_format=self.data_format, initializer=self.initializer,
                                              float_type=self.float_type)
                        x = utils.batch_norm('BatchNorm', x,
                                             trainable=True, data_format=self.data_format,
                                             mode=self.mode,
                                             use_gamma=self.bn_use_gamma, use_beta=self.bn_use_beta,
                                             bn_epsilon=self.bn_epsilon, bn_ema=self.bn_ema,
                                             float_type=self.float_type)
                        x = utils.relu(x)
                        if self.mode == 'train':
                            x = utils.dropout(x, keep_prob=0.9)

            print 'shape after block %d: ' % (block_index+1), x[0].get_shape()

        with tf.variable_scope('logits'):
            # new layers
            print 'logits: ', x[0].get_shape(),
            logits = utils.fully_connected(x, self.num_classes, trainable=True,
                                           data_format=self.data_format, initializer=self.initializer,
                                           float_type=self.float_type)

            self.logits = utils.resize_images(logits,
                                              self.image_shape[1:3] if self.data_format == 'NHWC'
                                              else self.image_shape[2:4],
                                              self.data_format,
                                              self.resize_images_method)
            print 'logits: ', self.logits[0].get_shape()
            self.probabilities = tf.nn.softmax(self.logits[0], dim=1 if self.data_format == 'NCHW' else 3)
            self.predictions = tf.argmax(self.logits[0], axis=1 if self.data_format == 'NCHW' else 3)

        print '================== network constructed ===================='
        return self.logits

    def _normal_loss(self, logits, labels):
        print 'normal cross entropy with softmax ... '
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_sum(xent)

    def _focal_loss_1(self, logits, labels, gamma=2):
        print 'focal loss: cross entropy with softmax ... '
        proba = tf.nn.softmax(logits, dim=-1)
        if gamma == 2:
            proba_gamma = tf.square(1 - proba)
        else:
            proba_gamma = tf.pow(1-proba, gamma)
        _t = tf.multiply(proba_gamma, tf.one_hot(labels, self.num_classes))
        _t = - 1.0 * tf.multiply(_t, tf.log(proba))
        return tf.reduce_sum(_t)

    def compute_loss(self, list_labels, logits=None):
        self.labels = list_labels
        if logits is None:
            logits = self.logits

        total_loss = tf.convert_to_tensor(0.0, dtype=self.float_type)
        total_auxiliary_loss = tf.convert_to_tensor(0.0, dtype=self.float_type)
        num_valide_pixel = 0
        for i in range(len(list_labels)):
            with tf.device('/gpu:%d' % i):
                print 'logit size:', logits[i].get_shape()
                print 'label size:', list_labels[i].get_shape()

                logit = tf.reshape(logits[i], [-1, self.num_classes])
                label = tf.reshape(list_labels[i], [-1, ])
                indice = tf.squeeze(tf.where(tf.less_equal(label, self.num_classes - 1)), 1)
                logit = tf.gather(logit, indice)
                label = tf.cast(tf.gather(label, indice), tf.int32)
                num_valide_pixel += tf.shape(label)[0]

                if self.loss_type == 'normal':
                    loss = self._normal_loss(logit, label)
                elif self.loss_type == 'focal_1':
                    loss = self._focal_loss_1(logit, label)
                else:
                    loss = self._normal_loss(logit, label)

                total_loss += loss

                if self.has_aux_loss:
                    print 'auxiliaire_logits size:', self.auxiliary_x[i].get_shape()
                    aux_l = tf.reshape(self.auxiliary_x[i], [-1, self.num_classes])
                    aux_l = tf.gather(aux_l, indice)

                    if self.loss_type == 'normal':
                        auxiliary_loss = self._normal_loss(aux_l, label)
                    elif self.loss_type == 'focal_1':
                        auxiliary_loss = self._focal_loss_1(aux_l, label)
                    else:
                        auxiliary_loss = self._normal_loss(aux_l, label)

                    total_auxiliary_loss += auxiliary_loss

        num_valide_pixel = tf.cast(num_valide_pixel, tf.float32)
        self.loss = tf.divide(total_loss, num_valide_pixel)
        if self.has_aux_loss:
            self.auxiliary_loss = tf.divide(total_auxiliary_loss, num_valide_pixel)
            self.auxiliary_loss *= tf.convert_to_tensor(0.4, dtype=self.float_type)

            self.loss += self.auxiliary_loss

        self.wd = 0
        if self.mode == 'train':
            self.wd = self._decay(self.wd_mode)

        return self.loss + self.wd

    def build_train_op(self, labels, logits=None):
        if self.train_like_in_paper:
            self.cost = self.compute_loss(labels, logits)
            if self.optimizer == 'sgd':
                print 'Applying Gradient Descent Optimizer...'
                opt_existing = tf.train.GradientDescentOptimizer(self.lrn_rate_placeholder)
                opt_new_norm = tf.train.GradientDescentOptimizer(self.lrn_rate_placeholder * 10)
                opt_new_bias = tf.train.GradientDescentOptimizer(self.lrn_rate_placeholder * 20)
            elif self.optimizer == 'mom':
                print 'Applying Momentum Optimizer...'
                opt_existing = tf.train.MomentumOptimizer(self.lrn_rate_placeholder, self.momentum)
                opt_new_norm = tf.train.MomentumOptimizer(self.lrn_rate_placeholder * 10, self.momentum)
                opt_new_bias = tf.train.MomentumOptimizer(self.lrn_rate_placeholder * 20, self.momentum)
            else:
                print 'unknown optimizer name: ', self.optimizer
                print 'Default to Momentum.'
                opt_existing = tf.train.MomentumOptimizer(self.lrn_rate_placeholder, self.momentum)
                opt_new_norm = tf.train.MomentumOptimizer(self.lrn_rate_placeholder * 10, self.momentum)
                opt_new_bias = tf.train.MomentumOptimizer(self.lrn_rate_placeholder * 20, self.momentum)

            existing_weights, new_normal_weights, new_bias_weights = self.get_different_variables()

            for v in existing_weights:
                print 'existing weights name: ', v.name
            for v in new_normal_weights:
                print 'new weights name: ', v.name
            for v in new_bias_weights:
                print 'new bias name: ', v.name

            grads = tf.gradients(self.cost, existing_weights + new_normal_weights + new_bias_weights,
                                 colocate_gradients_with_ops=True)

            grads_existing = grads[:len(existing_weights)]
            grads_new_norm = grads[len(existing_weights): (len(existing_weights) + len(new_normal_weights))]
            grads_new_bias = grads[(len(existing_weights) + len(new_normal_weights)):]

            train_existing = opt_existing.apply_gradients(zip(grads_existing, existing_weights))
            train_new_norm = opt_new_norm.apply_gradients(zip(grads_new_norm, new_normal_weights))
            train_new_bias = opt_new_bias.apply_gradients(zip(grads_new_bias, new_bias_weights))

            apply_op = tf.group(train_existing, train_new_norm, train_new_bias)

            train_ops = [apply_op] + self._extra_train_ops
            self.train_op = tf.group(*train_ops)
            return self.train_op
        else:
            return super(PSPNetMG, self).build_train_op(labels, logits)

    def get_different_variables(self):
        existing_weights = []
        new_normal_weights = []
        new_bias_weights = []
        for v in tf.trainable_variables():
            if any(elem in v.name for elem in self.new_layers_names):
                if 'bias' in v.name:
                    new_bias_weights.append(v)
                else:
                    new_normal_weights.append(v)
                continue
            existing_weights.append(v)

        for v in existing_weights:
            print 'existing weights name: ', v.name
        for v in new_normal_weights:
            print 'new weights name: ', v.name
        for v in new_bias_weights:
            print 'new bias name: ', v.name

        return existing_weights, new_normal_weights, new_bias_weights
