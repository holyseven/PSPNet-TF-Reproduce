import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages


def stride_arr(stride, data_format):
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    if data_format == 'NCHW':
        return [1, 1, stride, stride]
    else:  # NHWC
        return [1, stride, stride, 1]


def input_data(images, data_format):
    """
    images are alwyas NHWC.
    """
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    if data_format == 'NCHW':
        return tf.transpose(images, [0, 3, 1, 2])
    else:
        return images


def resize_images(feature_maps, output_size, data_format, method='bilinear'):
    if data_format == 'NCHW':
        _x = tf.transpose(feature_maps, [0, 2, 3, 1])
        if method == 'nn':
            _x = tf.image.resize_nearest_neighbor(_x, output_size)
        else:
            # output always tf.float32.
            _x = tf.image.resize_bilinear(_x, output_size)
        _x = tf.transpose(_x, [0, 3, 1, 2])
        return _x
    else:
        if method == 'nn':
            return tf.image.resize_nearest_neighbor(feature_maps, output_size)
        else:
            # output always tf.float32.
            return tf.image.resize_bilinear(feature_maps, output_size)


def max_pool(x, ksize, stride, data_format):
    return tf.nn.max_pool(x,
                          ksize=stride_arr(ksize, data_format),
                          strides=stride_arr(stride, data_format),
                          padding='SAME',
                          data_format=data_format)


def avg_pool(x, ksize, stride, data_format):
    return tf.nn.avg_pool(x,
                          ksize=stride_arr(ksize, data_format),
                          strides=stride_arr(stride, data_format),
                          padding='SAME',
                          data_format=data_format)


def global_avg_pool(x, data_format):
    assert x.get_shape().ndims == 4
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    if data_format == 'NCHW':
        return tf.reduce_mean(x, [2, 3], keep_dims=True)
    else:  # NHWC
        return tf.reduce_mean(x, [1, 2], keep_dims=True)


def conv2d_same(inputs, out_channels, kernel_size, stride, trainable,
                rate=1, data_format='NHWC', initializer='he', scope=None, float_type=tf.float32,
                he_init_std=None):
    # ======================== Checking valid values =========================
    if initializer not in ['he', 'xavier']:
        raise TypeError(
            "Only two initializers are supported at this moment: 'he' or 'xavier', "
            "%s is an unknown initializer." % initializer)
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    if not isinstance(stride, int):
        raise TypeError("Expecting an int for stride but %s is got." % stride)
    assert inputs.get_shape().ndims == 4, 'inputs should have rank 4.'
    # assert inputs.dtype == float_type, 'inputs data type %s is different from %s' % (inputs.dtype, float_type)

    # ======================== Setting default values =========================
    in_channels = inputs.get_shape().as_list()[-1]
    if data_format is 'NCHW':
        in_channels = inputs.get_shape().as_list()[1]

    initializer = tf.contrib.layers.xavier_initializer()
    if initializer is 'he':
        if he_init_std is None:
            n = kernel_size * kernel_size * out_channels
            std = np.sqrt(2.0 / n)
        else:
            std = he_init_std
        initializer = tf.random_normal_initializer(stddev=std)

    if scope is None:
        scope = 'weights'

    # ======================== Main operations =============================
    with tf.variable_scope(scope):
        kernel = tf.get_variable(
            '', [kernel_size, kernel_size, in_channels, out_channels],
            initializer=initializer,
            trainable=trainable,
            dtype=float_type
        )
        if stride == 1:
            if rate > 1:
                if data_format == 'NCHW':
                    inputs = tf.transpose(inputs, [0, 2, 3, 1])
                inputs = tf.nn.atrous_conv2d(inputs, kernel, rate, 'SAME')
                if data_format == 'NCHW':
                    inputs = tf.transpose(inputs, [0, 3, 1, 2])
                return inputs
            else:
                return tf.nn.conv2d(inputs, kernel, stride_arr(stride, data_format), 'SAME', data_format=data_format)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            if rate > 1:
                if data_format == 'NCHW':
                    inputs = tf.transpose(inputs, [0, 2, 3, 1])
                inputs = tf.nn.atrous_conv2d(inputs, kernel, rate, 'SAME')
                if data_format == 'NCHW':
                    inputs = tf.transpose(inputs, [0, 3, 1, 2])
                return inputs
            else:
                return tf.nn.conv2d(inputs, kernel, stride_arr(stride, data_format), 'VALID', data_format=data_format)


def batch_norm(name, inputs, trainable, data_format, mode,
               use_gamma=True, use_beta=True, bn_epsilon=1e-5, bn_ema=0.9, float_type=tf.float32):
    # This is a rapid version of batch normalization but it does not suit well for multiple gpus.
    # When trainable and not training mode, statistics will be frozen and parameters can be trained.

    def get_bn_variables(n_out, use_scale, use_bias, trainable, float_type):
        # TODO: not sure what to do.
        float_type = tf.float32

        if use_bias:
            beta = tf.get_variable('beta', [n_out],
                                   initializer=tf.constant_initializer(), trainable=trainable, dtype=float_type)
        else:
            beta = tf.zeros([n_out], name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [n_out],
                                    initializer=tf.constant_initializer(1.0), trainable=trainable, dtype=float_type)
        else:
            gamma = tf.ones([n_out], name='gamma')
        # x * gamma + beta

        moving_mean = tf.get_variable('moving_mean', [n_out],
                                      initializer=tf.constant_initializer(), trainable=False, dtype=float_type)
        moving_var = tf.get_variable('moving_variance', [n_out],
                                     initializer=tf.constant_initializer(1), trainable=False, dtype=float_type)
        return beta, gamma, moving_mean, moving_var

    def update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay):
        from tensorflow.contrib.framework import add_model_variable
        # TODO is there a way to use zero_debias in multi-GPU?
        update_op1 = moving_averages.assign_moving_average(
            moving_mean, batch_mean, decay, zero_debias=False,
            name='mean_ema_op')
        update_op2 = moving_averages.assign_moving_average(
            moving_var, batch_var, decay, zero_debias=False,
            name='var_ema_op')
        add_model_variable(moving_mean)
        add_model_variable(moving_var)

        # seems faster than delayed update, but might behave otherwise in distributed settings.
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')

    # ======================== Checking valid values =========================
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    assert inputs.get_shape().ndims == 4, 'inputs should have rank 4.'
    assert inputs.dtype == float_type, 'inputs data type is different from %s' % float_type
    if mode not in ['train', 'training', 'val', 'validation', 'test', 'eval']:
        raise TypeError("Unknown mode.")

    # ======================== Setting default values =========================
    shape = inputs.get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]
    if data_format == 'NCHW':
        n_out = shape[1]
    if mode is 'training' or mode is 'train':
        mode = 'train'
    else:
        mode = 'test'

    # ======================== Main operations =============================
    with tf.variable_scope(name):
        beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_gamma, use_beta, trainable, float_type)

        if mode == 'train' and trainable:
            xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
                inputs, gamma, beta, epsilon=bn_epsilon,
                is_training=True, data_format=data_format)
            if tf.get_variable_scope().reuse:
                return xn
            else:
                return update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, bn_ema)
        else:
            xn = tf.nn.batch_normalization(
                inputs, moving_mean, moving_var, beta, gamma, bn_epsilon)
            return xn


def batch_norm_from_layers(name, inputs, trainable, data_format, mode,
                           use_gamma=True, use_beta=True, bn_epsilon=1e-5, bn_ema=0.9):
    from tensorflow.contrib.layers import batch_norm as bn

    # if using this, should be note that:
    # python
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.minimize(loss)

    return bn(inputs, decay=bn_ema, center=use_gamma, scale=use_beta, epsilon=bn_epsilon,
              is_training=(mode=='train'), trainable=trainable, scope=name, data_format=data_format)


def relu(inputs):
    return tf.nn.relu(inputs, name='relu')


def identity_mapping_v1(inputs, out_channels, stride, data_format, initializer='he', rate=1, trainable=True,
                        bn_mode='train', bn_use_gamma=True, bn_use_beta=True, bn_epsilon=1e-5, bn_ema=0.9,
                        float_type=tf.float32):
    in_channels = inputs.get_shape().as_list()[-1]
    if data_format is 'NCHW':
        in_channels = inputs.get_shape().as_list()[1]

    orig_x = inputs
    with tf.variable_scope('identity_mapping_v1/conv1'):
        x = conv2d_same(inputs, out_channels, 3, stride, trainable, rate, data_format, initializer,
                        float_type=float_type)
        x = batch_norm('BatchNorm', x, trainable, data_format, bn_mode, bn_use_gamma, bn_use_beta, bn_epsilon,
                       bn_ema, float_type)
        x = relu(x)

    with tf.variable_scope('identity_mapping_v1/conv2'):
        x = conv2d_same(x, out_channels, 3, 1, trainable, rate, data_format, initializer, float_type=float_type)
        x = batch_norm('BatchNorm', x, trainable, data_format, bn_mode, bn_use_gamma, bn_use_beta, bn_epsilon,
                       bn_ema, float_type)

    with tf.variable_scope('identity_mapping_v1/add'):
        if in_channels != out_channels:
            orig_x = avg_pool(orig_x, stride, stride, data_format)
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_channels-in_channels)//2, (out_channels-in_channels)//2]])
        x += orig_x
        x = relu(x)

    return x


def bottleneck_residual(inputs, out_channels, stride, data_format,
                        initializer='he', rate=1, trainable=True,
                        bn_mode='train', bn_use_gamma=True, bn_use_beta=True, bn_epsilon=1e-5, bn_ema=0.9,
                        float_type=tf.float32):
    """Bottleneck v1 residual unit with 3 sub layers."""

    # ======================== Checking valid values =========================
    if initializer not in ['he', 'xavier']:
        raise TypeError(
            "Only two initializers are supported at this moment: 'he' or 'xavier', "
            "%s is an unknown initializer." % initializer)
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    if not isinstance(stride, int):
        raise TypeError("Expecting an int for stride but %s is got." % stride)
    assert inputs.get_shape().ndims == 4, 'inputs should have rank 4.'
    # ======================== Setting default values =========================
    in_channels = inputs.get_shape().as_list()[-1]
    if data_format is 'NCHW':
        in_channels = inputs.get_shape().as_list()[1]

    # ======================== Main operations =============================
    orig_x = inputs
    with tf.variable_scope('bottleneck_v1/conv1'):
        """1x1, 64->64"""
        x = conv2d_same(inputs, out_channels / 4, 1, 1,
                        trainable=trainable, data_format=data_format, initializer=initializer, float_type=float_type)
        x = batch_norm('BatchNorm', x, trainable, data_format, bn_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema,
                       float_type)
        x = relu(x)

    with tf.variable_scope('bottleneck_v1/conv2'):
        """3x3, 64->64"""
        x = conv2d_same(x, out_channels / 4, 3, stride, trainable, rate, data_format, initializer,
                        float_type=float_type)
        x = batch_norm('BatchNorm', x, trainable, data_format, bn_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema,
                       float_type)
        x = relu(x)

    with tf.variable_scope('bottleneck_v1/conv3'):
        """1x1, 64->256"""
        x = conv2d_same(x, out_channels, 1, 1,
                        trainable=trainable, data_format=data_format, initializer=initializer, float_type=float_type)
        x = batch_norm('BatchNorm', x, trainable, data_format, bn_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema,
                       float_type)

    """1x1, 64->256 or 256==256, do nothing."""
    if in_channels != out_channels:
        with tf.variable_scope('bottleneck_v1/shortcut'):
            orig_x = conv2d_same(orig_x, out_channels, 1, stride,
                                 trainable=trainable, data_format=data_format,
                                 initializer=initializer, float_type=float_type)
            orig_x = batch_norm('BatchNorm', orig_x, trainable, data_format,
                                bn_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema, float_type)
    else:
        if stride > 1:
            orig_x = max_pool(orig_x, 1, 2, data_format)

    x += orig_x
    x = relu(x)
    return x


def subsample(inputs, stride, data_format='NHWC'):
    if stride == 1:
        return inputs
    else:
        return max_pool(inputs, 1, stride, data_format)


def bottleneck_residual_v2(inputs, out_channels, stride, data_format,
                           initializer='he', rate=1, trainable=True,
                           bn_mode='train', bn_use_gamma=True, bn_use_beta=True, bn_epsilon=1e-5, bn_ema=0.9,
                           float_type=tf.float32):
    """Bottleneck v2 residual unit with 3 sub layers."""

    # ======================== Checking valid values =========================
    if initializer not in ['he', 'xavier']:
        raise TypeError(
            "Only two initializers are supported at this moment: 'he' or 'xavier', "
            "%s is an unknown initializer." % initializer)
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    if not isinstance(stride, int):
        raise TypeError("Expecting an int for stride but %s is got." % stride)
    assert inputs.get_shape().ndims == 4, 'inputs should have rank 4.'
    # ======================== Setting default values =========================
    in_channels = inputs.get_shape().as_list()[-1]
    if data_format is 'NCHW':
        in_channels = inputs.get_shape().as_list()[1]

    # ======================== Main operations =============================
    with tf.variable_scope('bottleneck_v2'):
        preact = batch_norm('preact', inputs, trainable, data_format,
                            bn_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema, float_type)
        preact = relu(preact)

        with tf.variable_scope('shortcut'):
            if out_channels == in_channels:
                shortcut = subsample(inputs, stride, data_format)
            else:
                shortcut = conv_bias_relu(preact, out_channels, 1, stride, trainable,
                                          relu=False, data_format=data_format,
                                          initializer=initializer, float_type=float_type, bias_scope='biases')

        with tf.variable_scope('conv1'):
            residual = conv2d_same(preact, out_channels / 4, 1, 1, trainable, 1, data_format, initializer,
                                   float_type=float_type)
            residual = batch_norm('BatchNorm', residual, trainable, data_format, bn_mode, bn_use_gamma, bn_use_beta,
                                  bn_epsilon, bn_ema, float_type)
            residual = relu(residual)
        with tf.variable_scope('conv2'):
            residual = conv2d_same(residual, out_channels / 4, 3, stride, trainable, rate, data_format, initializer,
                                   float_type=float_type)
            residual = batch_norm('BatchNorm', residual, trainable, data_format, bn_mode, bn_use_gamma, bn_use_beta,
                                  bn_epsilon, bn_ema, float_type)
            residual = relu(residual)
        with tf.variable_scope('conv3'):
            residual = conv_bias_relu(residual, out_channels, 1, 1, trainable, relu=False,
                                      data_format=data_format, initializer=initializer,
                                      float_type=float_type, bias_scope='biases')

        output = shortcut + residual

    return output


def fully_connected(inputs, out_channels, trainable=True, data_format='NHWC', initializer='he', float_type=tf.float32):
    """convolution 1x1 layer for final output."""
    x = conv2d_same(inputs, out_channels, 1, 1,
                    trainable=trainable, data_format=data_format, initializer=initializer, float_type=float_type)
    b = tf.get_variable('biases', [out_channels], trainable=trainable,
                        initializer=tf.constant_initializer(0.01), dtype=float_type)
    return tf.nn.bias_add(x, b, data_format=data_format)


def conv_bias_relu(x, out_channels, kernel_size, stride,
                   trainable=True, relu=True, data_format='NHWC', initializer='he', float_type=tf.float32,
                   bias_scope='bias'):

    x = conv2d_same(x, out_channels, kernel_size, stride, trainable,
                    data_format=data_format,
                    initializer=initializer,
                    float_type=float_type)

    bias = tf.get_variable(bias_scope, [out_channels], initializer=tf.constant_initializer(0.01))
    x = tf.nn.bias_add(x, bias, data_format=data_format)
    if relu:
        x = tf.nn.relu(x)

    return x


def conv2d_transpose(name, x, out_channels,
                  ksize=4, stride=2, data_format='NHWC', trainable=False):
    def get_transpose_filter(weights_shape, trainable):
        """
        This seems to be a bilinear interpolation implementation.
        """
        # TODO: check what is the difference between this and resize_images.
        # weights_shape: [height, width, output_channels, in_channels]
        from math import ceil
        import numpy as np
        width = weights_shape[0]
        height = weights_shape[1]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([weights_shape[0], weights_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(weights_shape)
        for i in range(weights_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name='weights', initializer=init,
                              shape=weights.shape, trainable=trainable)
        return var

    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if data_format == 'NHWC':
            in_features = x.get_shape()[3].value
        else:  # NCHW
            in_features = x.get_shape()[1].value

        # Compute shape out of Bottom
        in_shape = tf.shape(x)

        if data_format == 'NHWC':
            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, out_channels]
        else:  # NCHW
            h = ((in_shape[2] - 1) * stride) + 1
            w = ((in_shape[3] - 1) * stride) + 1
            new_shape = [in_shape[0], out_channels, h, w]

        output_shape = tf.stack(new_shape)

        weights_shape = [ksize, ksize, out_channels, in_features]

        weights = get_transpose_filter(weights_shape, trainable)
        if trainable:
            print 'training conv2d_transpose layer: ', name
        deconv = tf.nn.conv2d_transpose(x, weights, output_shape,
                                        strides=strides, padding='SAME', data_format=data_format)

    return deconv


def dense_crf(probs, img=None, n_iters=10, n_classes=19,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13)):
    import pydensecrf.densecrf as dcrf
    _, h, w, _ = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
    U = -np.log(probs)  # Unary potential.
    U = U.reshape((n_classes, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    if img is not None:
        assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)