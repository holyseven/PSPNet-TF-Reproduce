
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
        list_conv2dt = tf.get_collection('init_conv2dt_weights')

        if mode == 0:
            print 'Applying L2 regularization...'
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    l2_losses_existing_layers += tf.nn.l2_loss(v)
            return tf.multiply(self.wd_rate_placeholder, l2_losses_existing_layers) \
                   + tf.multiply(self.wd_rate_placeholder2, l2_losses_new_layers)
        elif mode == 1:
            print 'Applying L2-SP regularization...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue

                    print v.name

                    name = v.name.split(':')[0]
                    if reader.has_tensor(name):
                        pre_trained_weights = reader.get_tensor(name)
                    else:
                        name = name.split('/weights')[0]
                        for elem in list_conv2dt:
                            if elem.name.split('/Const')[0] == name:
                                pre_trained_weights = elem
                                break
                        # print v, pre_trained_weights

                    l2_losses_existing_layers += tf.nn.l2_loss(v - pre_trained_weights)
            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 10:
            print 'Applying L2-SP regularization for weights and gammas...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name or 'gamma' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    print v.name

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    l2_losses_existing_layers += tf.nn.l2_loss(v - pre_trained_weights)

            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 11:
            print 'Applying L2-SP variant which computes the negative cosine similarity -cos(w0,w)...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    print v.name

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    pre_trained_weights_norm = np.sqrt(np.sum(np.square(pre_trained_weights)))

                    cos_loss = tf.reduce_sum(tf.multiply(v, pre_trained_weights / pre_trained_weights_norm))
                    cos_loss = - cos_loss / tf.norm(v)

                    # cos_loss = - cos(pre_trained_weights, v)
                    l2_losses_existing_layers += cos_loss

            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 12:
            print 'Applying L2-SP variant which computes the inverse of cosine similarity 1/cos(w0,w)...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = []
            l2_losses_new_layers = []
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    print v.name

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    pre_trained_weights_norm = np.sqrt(np.sum(np.square(pre_trained_weights)))

                    cos_loss = tf.reduce_sum(tf.multiply(v, pre_trained_weights / pre_trained_weights_norm))
                    cos_loss = tf.norm(v) / cos_loss

                    # cos_loss = - cos(pre_trained_weights, v)
                    l2_losses_existing_layers += cos_loss

            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 101:
            print 'Applying L2-SP regularization for all parameters (weights, biases, gammas, betas)...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                name = v.name.split(':')[0]
                pre_trained_weights = reader.get_tensor(name)

                if any(elem in v.name for elem in self.new_layers_names):
                    print 'except ', v.name
                    l2_losses_new_layers += tf.nn.l2_loss(v)
                    continue
                l2_losses_existing_layers += tf.nn.l2_loss(v - pre_trained_weights)
            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 2:
            print 'Applying L2-SP-k regularization...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = []
            l2_losses_new_layers = []
            weights_varianbles = []

            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    # I know there is only four new layers with 'weights'
                    # and they are named by 'fc1_voc12_c?/weights:0' ? = 0, 1, 2, 3
                    if 'logits' in v.name:
                        l2_losses_new_layers.append(tf.nn.l2_loss(v))
                        print 'new layers', v.name
                        continue

                    weights_varianbles.append(v)

            for i in range(len(weights_varianbles)):
                name = weights_varianbles[i].name.split(':')[0]

                pre_trained_weights = reader.get_tensor(name)
                single_loss = tf.nn.l2_loss(weights_varianbles[i] - pre_trained_weights)

                if 'shortcut' in name:
                    # because these layers are parallel to another three layers.
                    l2_losses_existing_layers.append(tf.scalar_mul(len(weights_varianbles) - i + 3, single_loss))
                    print 'existing layers: ', len(weights_varianbles) - i + 3, name
                else:
                    l2_losses_existing_layers.append(tf.scalar_mul(len(weights_varianbles) - i, single_loss))
                    print 'existing layers: ', len(weights_varianbles) - i, name

            return self.wd_rate_placeholder * tf.add_n(l2_losses_existing_layers) \
                   + self.wd_rate_placeholder2 * tf.add_n(l2_losses_new_layers)
        elif mode == 3:
            print 'Applying L2-SP-exp regularization...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l2_losses_existing_layers = []
            l2_losses_new_layers = []
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)

                    if 'logits' in v.name:
                        print 'except ', v.name
                        l2_losses_new_layers.append(tf.nn.l2_loss(v))
                        continue
                    dif = v - pre_trained_weights
                    l2_losses_existing_layers.append(tf.reduce_sum(tf.exp(tf.multiply(dif, dif))))
            return self.wd_rate_placeholder * tf.add_n(l2_losses_existing_layers) \
                   + self.wd_rate_placeholder2 * tf.add_n(l2_losses_new_layers)
        elif mode == 4:
            print 'Applying L1-SP regularization...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l1_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)

                    l1_losses_existing_layers += tf.reduce_sum(tf.abs(v - pre_trained_weights))
            return self.wd_rate_placeholder * l1_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 41:
            print 'Applying L1-SP regularization with sqrt(x^2+\epsilon)...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            l1_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue
                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    l1_losses_existing_layers += tf.reduce_sum(tf.sqrt((v - pre_trained_weights) ** 2 + 1e-16))

            return self.wd_rate_placeholder * l1_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 5:
            print 'Applying Group-Lasso-SP regularization...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)

            lasso_loss_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    v_shape = v.get_shape().as_list()
                    assert len(v_shape) == 4

                    # split for each output feature map.
                    sum = tf.reduce_sum((v - pre_trained_weights) ** 2, axis=[0, 1, 2]) + 1e-16
                    alpha_k = float(np.sqrt(v_shape[0] * v_shape[1] * v_shape[2]))

                    # if 'resnet_v1_101/conv1/weights' in v.name:
                    #     self.sum_1 = sum
                    # if 'resnet_v1_101/block2/unit_4/bottleneck_v1/conv1/weights' in v.name:
                    #     self.sum_2 = sum

                    sqrt = tf.reduce_sum(tf.sqrt(sum))
                    lasso_loss_existing_layers += sqrt * alpha_k

            return self.wd_rate_placeholder * lasso_loss_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 50:
            print 'Applying Group-Lasso-SP regularization, each group is one whole layer...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)

            lasso_loss_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                        continue

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    v_shape = v.get_shape().as_list()
                    assert len(v_shape) == 4

                    # split for each output feature map.
                    sum = tf.reduce_sum((v - pre_trained_weights) ** 2, axis=[0, 1, 2, 3]) + 1e-16
                    alpha_k = float(np.sqrt(v_shape[0] * v_shape[1] * v_shape[2] * v_shape[3]))

                    # if 'resnet_v1_101/conv1/weights' in v.name:
                    #     self.sum_1 = sum
                    # if 'resnet_v1_101/block2/unit_4/bottleneck_v1/conv1/weights' in v.name:
                    #     self.sum_2 = sum

                    sqrt = tf.reduce_sum(tf.sqrt(sum))
                    lasso_loss_existing_layers += sqrt * alpha_k

            return self.wd_rate_placeholder * lasso_loss_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 6:
            print 'Applying L2-SP-Fisher Fisher Information Matrix (FIM) + fisher_epsilon,', self.fisher_epsilon, '...'

            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            fim_dict = np.load(self.fisher_filename).item()
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    fim = fim_dict[v.name] + self.fisher_epsilon
                    # print fim.shape, v.get_shape()

                    l2_losses_existing_layers += tf.reduce_sum(0.5 * fim * tf.square(v-pre_trained_weights))
            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 61:
            print 'Applying L2-SP-Fisher regularizatino clip(FIM) with max value', self.fisher_epsilon, '...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)
            fim_dict = np.load(self.fisher_filename).item()
            l2_losses_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)
                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    fim = np.clip(fim_dict[v.name], a_min=None, a_max=self.fisher_epsilon)
                    # print fim.shape, v.get_shape()

                    l2_losses_existing_layers += tf.reduce_sum(0.5 * fim * tf.square(v-pre_trained_weights))
            return self.wd_rate_placeholder * l2_losses_existing_layers \
                   + self.wd_rate_placeholder2 * l2_losses_new_layers
        elif mode == 7:
            print 'Applying Group-Lasso-SP-Fisher regularization + fisher_epsilon,', self.fisher_epsilon, '...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)

            fim_dict = np.load(self.fisher_filename).item()
            lasso_loss_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    fim = fim_dict[v.name] + self.fisher_epsilon

                    v_shape = v.get_shape().as_list()
                    assert len(v_shape) == 4

                    # split for each output feature map.
                    a = tf.multiply((v - pre_trained_weights) ** 2, fim)
                    sum = tf.reduce_sum(a, axis=[0, 1, 2]) + 1e-16
                    alpha_k = float(np.sqrt(v_shape[0] * v_shape[1] * v_shape[2]))

                    sqrt = tf.reduce_sum(tf.sqrt(sum))
                    lasso_loss_existing_layers += sqrt * alpha_k

            return self.wd_rate_placeholder * lasso_loss_existing_layers + self.wd_rate_placeholder2 * l2_losses_new_layers

        elif mode == 71:
            print 'Applying Group-Lasso-SP-Fisher regularizatino clip(FIM) with max value', self.fisher_epsilon, '...'
            reader = tf.train.NewCheckpointReader(self.fine_tune_filename)

            fim_dict = np.load(self.fisher_filename).item()
            lasso_loss_existing_layers = 0.0
            l2_losses_new_layers = 0.0
            for v in tf.trainable_variables():
                if 'weights' in v.name:
                    if any(elem in v.name for elem in self.new_layers_names):
                        print 'except ', v.name
                        l2_losses_new_layers += tf.nn.l2_loss(v)

                    name = v.name.split(':')[0]
                    pre_trained_weights = reader.get_tensor(name)
                    fim = np.clip(fim_dict[v.name], a_min=None, a_max=self.fisher_epsilon)

                    v_shape = v.get_shape().as_list()
                    assert len(v_shape) == 4

                    # split for each output feature map.
                    a = tf.multiply((v - pre_trained_weights) ** 2, fim)
                    sum = tf.reduce_sum(a, axis=[0, 1, 2]) + 1e-16
                    alpha_k = float(np.sqrt(v_shape[0] * v_shape[1] * v_shape[2]))

                    sqrt = tf.reduce_sum(tf.sqrt(sum))
                    lasso_loss_existing_layers += sqrt * alpha_k

            return self.wd_rate_placeholder * lasso_loss_existing_layers + self.wd_rate_placeholder2 * l2_losses_new_layers

        print 'No regularization...'
        return tf.convert_to_tensor(0.0)

