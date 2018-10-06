"""
mg means multiple gpus. The idea here is very simple: changing the inputs and outputs to a list whose length is equal
to the number of gpus. All operations are defined under `with tf.device('/gpu:%d' % i):`.
Except batch normalization, all functions have no difference from the implementation in a single gpu.
"""
from __future__ import print_function, division, absolute_import
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


def input_data(list_images, data_format):
    """
    images are alwyas NHWC.
    """
    assert type(list_images) == list
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    if data_format == 'NHWC':
        return list_images
    else:
        list_output = []
        for i in range(len(list_images)):
            with tf.device('/gpu:%d' % i):
                list_output.append(tf.transpose(list_images[i], [0, 3, 1, 2]))
        return list_output


def resize_images(list_images, output_size, data_format, method='bilinear', train_conv2dt=False, stride_conv2dt=8):
    assert type(list_images) == list

    list_output = []
    if train_conv2dt:
        return conv2d_transpose('conv_transpose', list_images,
                                ksize=stride_conv2dt // 2,
                                stride=stride_conv2dt,
                                data_format=data_format)

    for i in range(len(list_images)):
        with tf.device('/gpu:%d' % i):
            if data_format == 'NCHW':
                _x = tf.transpose(list_images[i], [0, 2, 3, 1])
                if method == 'nn':
                    _x = tf.image.resize_nearest_neighbor(_x, output_size)
                elif method == 'area':
                    _x = tf.image.resize_area(_x, output_size)
                elif method == 'cubic':
                    _x = tf.image.resize_bicubic(_x, output_size)
                else:
                    # output always tf.float32.
                    _x = tf.image.resize_bilinear(_x, output_size)
                _x = tf.transpose(_x, [0, 3, 1, 2])
                list_output.append(_x)
            else:
                if method == 'nn':
                    _x = tf.image.resize_nearest_neighbor(list_images[i], output_size)
                elif method == 'area':
                    _x = tf.image.resize_area(list_images[i], output_size)
                elif method == 'cubic':
                    _x = tf.image.resize_bicubic(list_images[i], output_size)
                else:
                    # output always tf.float32.
                    _x = tf.image.resize_bilinear(list_images[i], output_size)
                list_output.append(_x)

    return list_output


def max_pool(list_input, ksize, stride, data_format):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            _x = tf.nn.max_pool(list_input[i],
                                ksize=stride_arr(ksize, data_format),
                                strides=stride_arr(stride, data_format),
                                padding='SAME',
                                data_format=data_format)
            list_output.append(_x)
    return list_output


def global_avg_pool(list_input, data_format):
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            if data_format == 'NCHW':
                _x = tf.reduce_mean(list_input[i], [2, 3], keep_dims=True)
            else:  # NHWC
                _x = tf.reduce_mean(list_input[i], [1, 2], keep_dims=True)
            list_output.append(_x)
    return list_output


def conv2d_same(list_input, out_channels, kernel_size, stride, trainable=True,
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
    assert type(list_input) == list
    # ======================== Setting default values =========================
    in_channels = list_input[0].get_shape().as_list()[-1]
    if data_format == 'NCHW':
        in_channels = list_input[0].get_shape().as_list()[1]

    initializer = tf.contrib.layers.xavier_initializer()
    if initializer == 'he':
        if he_init_std is None:
            n = kernel_size * kernel_size * out_channels
            std = np.sqrt(2.0 / n)
        else:
            std = he_init_std
        initializer = tf.random_normal_initializer(stddev=std)

    if scope is None:
        scope = 'weights'

    # ======================== Main operations =============================

    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(scope, reuse=(i>0)):
                kernel = tf.get_variable(
                    '', [kernel_size, kernel_size, in_channels, out_channels],
                    initializer=initializer,
                    trainable=trainable,
                    dtype=float_type
                )
                if stride == 1:
                    if rate > 1:
                        if data_format == 'NCHW':
                            _x = tf.transpose(list_input[i], [0, 2, 3, 1])
                        else:
                            _x = tf.convert_to_tensor(list_input[i])
                        _x = tf.nn.atrous_conv2d(_x, kernel, rate, 'SAME')
                        if data_format == 'NCHW':
                            _x = tf.transpose(_x, [0, 3, 1, 2])
                        list_output.append(_x)
                    else:
                        _x = tf.nn.conv2d(list_input[i], kernel, stride_arr(stride, data_format), 'SAME',
                                          data_format=data_format)
                        list_output.append(_x)
                else:
                    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
                    pad_total = kernel_size_effective - 1
                    pad_beg = pad_total // 2
                    pad_end = pad_total - pad_beg
                    if data_format == 'NCHW':
                        inputs = tf.pad(list_input[i],
                                        [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
                    else:
                        inputs = tf.pad(list_input[i],
                                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
                    if rate > 1:
                        if data_format == 'NCHW':
                            inputs = tf.transpose(inputs, [0, 2, 3, 1])
                        _x = tf.nn.atrous_conv2d(inputs, kernel, rate, 'SAME')
                        if data_format == 'NCHW':
                            _x = tf.transpose(_x, [0, 3, 1, 2])
                        list_output.append(_x)
                    else:
                        _x = tf.nn.conv2d(inputs, kernel, stride_arr(stride, data_format), 'VALID',
                                          data_format=data_format)
                        list_output.append(_x)
    return list_output


def batch_norm(list_input, stats_mode, data_format='NHWC', float_type=tf.float32, trainable=True,
               use_gamma=True, use_beta=True, bn_epsilon=1e-5, bn_ema=0.9, scope='BatchNorm'):

    def get_bn_variables(n_out, use_scale, use_bias, trainable, float_type):
        # TODO: not sure what to do.
        # float_type = tf.float32

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
    assert type(list_input) == list

    # ======================== Setting default values =========================
    shape = list_input[0].get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]
    if data_format == 'NCHW':
        n_out = shape[1]

    # ======================== Main operations =============================
    means = []
    square_means = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            batch_mean = tf.reduce_mean(list_input[i], [0, 1, 2])
            batch_square_mean = tf.reduce_mean(tf.square(list_input[i]), [0, 1, 2])
            means.append(batch_mean)
            square_means.append(batch_square_mean)

    with tf.device('/cpu:0'):  # if your GPUs have NVLinks and you've install NCCL2, you can change `/cpu:0` to `/gpu:0`
        shape = tf.shape(list_input[0])
        num = shape[0] * shape[1] * shape[2] * len(list_input)
        mean = tf.reduce_mean(means, axis=0)
        var = tf.reduce_mean(square_means, axis=0) - tf.square(mean)
        var *= tf.cast(num, float_type) / tf.cast(num-1, float_type)  # unbiased variance

    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(scope, reuse=i > 0):
                beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_gamma, use_beta,
                                                                        trainable, float_type)

                if 'train' in stats_mode:
                    xn = tf.nn.batch_normalization(
                        list_input[i], mean, var, beta, gamma, bn_epsilon)
                    if tf.get_variable_scope().reuse or 'gather' not in stats_mode:
                        list_output.append(xn)
                    else:
                        # gather stats and it is the main gpu device.
                        xn = update_bn_ema(xn, mean, var, moving_mean, moving_var, bn_ema)
                        list_output.append(xn)
                else:
                    xn = tf.nn.batch_normalization(
                        list_input[i], moving_mean, moving_var, beta, gamma, bn_epsilon)
                    list_output.append(xn)

    return list_output


def relu(list_input):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            output = tf.nn.relu(list_input[i], name='relu')
            list_output.append(output)
    return list_output


def bottleneck_residual(list_input, out_channels, stride,
                        initializer='he', rate=1, trainable=True, float_type=tf.float32, data_format='NHWC',
                        bn_mode='train_gather', bn_use_gamma=True, bn_use_beta=True, bn_epsilon=1e-5, bn_ema=0.9):
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
    assert type(list_input) == list
    # ======================== Setting default values =========================
    if stride > 1:
        rate = 1  # because of atrous_conv2d.

    in_channels = list_input[0].get_shape().as_list()[-1]
    if data_format is 'NCHW':
        in_channels = list_input[0].get_shape().as_list()[1]

    # ======================== Main operations =============================
    orig_x = list_input
    with tf.variable_scope('bottleneck_v1/conv1'):
        """1x1, 64->64"""
        x = conv2d_same(list_input, out_channels // 4, 1, 1,
                        trainable=trainable, data_format=data_format, initializer=initializer, float_type=float_type)
        x = batch_norm(x, bn_mode, data_format, float_type, trainable,
                       bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
        x = relu(x)

    with tf.variable_scope('bottleneck_v1/conv2'):
        """3x3, 64->64"""
        x = conv2d_same(x, out_channels // 4, 3, stride, trainable, rate, data_format, initializer,
                        float_type=float_type)
        x = batch_norm(x, bn_mode, data_format, float_type, trainable,
                       bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
        x = relu(x)

    with tf.variable_scope('bottleneck_v1/conv3'):
        """1x1, 64->256"""
        x = conv2d_same(x, out_channels, 1, 1,
                        trainable=trainable, data_format=data_format, initializer=initializer, float_type=float_type)
        x = batch_norm(x, bn_mode, data_format, float_type, trainable,
                       bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)

    """1x1, 64->256 or 256==256, do nothing."""
    if in_channels != out_channels:
        with tf.variable_scope('bottleneck_v1/shortcut'):
            orig_x = conv2d_same(orig_x, out_channels, 1, stride,
                                 trainable=trainable, data_format=data_format,
                                 initializer=initializer, float_type=float_type)
            orig_x = batch_norm(orig_x, bn_mode, data_format, float_type, trainable,
                                bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
    else:
        if stride > 1:
            orig_x = max_pool(orig_x, 1, 2, data_format)

    for i in range(len(x)):
        with tf.device('/gpu:%d' % i):
            x[i] += orig_x[i]

    x = relu(x)
    return x


def subsample(list_input, stride, data_format='NHWC'):
    if stride == 1:
        return list_input
    else:
        return max_pool(list_input, 1, stride, data_format)


def bottleneck_residual_v2(list_input, out_channels, stride, data_format,
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
    assert list_input[0].get_shape().ndims == 4, 'inputs should have rank 4.'
    # ======================== Setting default values =========================
    in_channels = list_input[0].get_shape().as_list()[-1]
    if data_format is 'NCHW':
        in_channels = list_input[0].get_shape().as_list()[1]

    # ======================== Main operations =============================
    with tf.variable_scope('bottleneck_v2'):
        preact = batch_norm(list_input, bn_mode, data_format, float_type, trainable,
                            bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema, scope='preact')
        preact = relu(preact)

        with tf.variable_scope('shortcut'):
            if out_channels == in_channels:
                shortcut = subsample(list_input, stride, data_format)
            else:
                shortcut = conv_bias_relu(preact, out_channels, 1, stride, trainable,
                                          relu=False, data_format=data_format,
                                          initializer=initializer, float_type=float_type, bias_scope='biases')

        with tf.variable_scope('conv1'):
            residual = conv2d_same(preact, out_channels // 4, 1, 1, trainable, 1, data_format, initializer,
                                   float_type=float_type)
            residual = batch_norm(residual, bn_mode, data_format, float_type, trainable,
                                  bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
            residual = relu(residual)
        with tf.variable_scope('conv2'):
            residual = conv2d_same(residual, out_channels // 4, 3, stride, trainable, rate, data_format, initializer,
                                   float_type=float_type)
            residual = batch_norm(residual, bn_mode, data_format, float_type, trainable,
                                  bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
            residual = relu(residual)
        with tf.variable_scope('conv3'):
            residual = conv_bias_relu(residual, out_channels, 1, 1, trainable, relu=False,
                                      data_format=data_format, initializer=initializer,
                                      float_type=float_type, bias_scope='biases')

        for i in range(len(residual)):
            with tf.device('/gpu:%d' % i):
                residual[i] += shortcut[i]

    return residual


def fully_connected(list_input, out_channels, trainable=True, data_format='NHWC', initializer='he', float_type=tf.float32):
    """convolution 1x1 layer for final output."""
    list_x = conv2d_same(list_input, out_channels, 1, 1,
                    trainable=trainable, data_format=data_format, initializer=initializer, float_type=float_type)

    list_output = []
    for i in range(len(list_x)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope('biases', reuse=(i > 0)):
                b = tf.get_variable('', [out_channels], initializer=tf.constant_initializer(0.01))
                output = tf.nn.bias_add(list_x[i], b, data_format)
                list_output.append(output)

    return list_output


def conv_bias_relu(list_input, out_channels, kernel_size, stride, trainable=True, relu=True,
                   data_format='NHWC', initializer='he', float_type=tf.float32, bias_scope='biases'):
    list_x = conv2d_same(list_input, out_channels, kernel_size, stride,
                         trainable=trainable, data_format=data_format, initializer=initializer, float_type=float_type)

    list_bias = []
    for i in range(len(list_x)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(bias_scope, reuse=(i > 0)):
                b = tf.get_variable('', [out_channels], initializer=tf.constant_initializer(0.01))
                output = tf.nn.bias_add(list_x[i], b, data_format)
                list_bias.append(output)
    if relu:
        list_output = tf.nn.relu(list_bias)
        return list_output
    else:
        return list_bias


def get_transpose_weights(weights_shape):
    # TODO: check what is the difference between this and resize_images. => it is trainable? the only difference?
    # weights_shape: [height, width, output_channels, in_channels]
    from math import ceil
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

    return weights


def conv2d_transpose(name, list_input, out_channels=None,
                     ksize=4, stride=2, data_format='NHWC', trainable=True):
    # TODO: This does not perform the same as resize_image. check the difference.
    assert type(list_input) == list
    strides = [1, stride, stride, 1]
    list_output = []

    if data_format == 'NHWC':
        in_features = list_input[0].get_shape()[3].value
    else:  # NCHW
        in_features = list_input[0].get_shape()[1].value

    if out_channels is None:
        out_channels = in_features

    # Compute shape out of Bottom
    in_shape = list_input[0].get_shape()

    if data_format == 'NHWC':
        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = [in_shape[0], h, w, out_channels]
    else:  # NCHW
        h = in_shape[2] * stride
        w = in_shape[3] * stride
        new_shape = [in_shape[0], out_channels, h, w]

    output_shape = tf.stack(new_shape)
    weights_shape = [ksize, ksize, out_channels, in_features]

    weights = get_transpose_weights(weights_shape)
    with tf.variable_scope(name):
        init_conv2dt_weights = tf.constant(weights, dtype=tf.float32)

    tf.add_to_collection('init_conv2dt_weights', init_conv2dt_weights)

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(name, reuse=(i>0)):
                var = tf.get_variable(name='weights', initializer=init,
                                      shape=weights.shape, trainable=trainable)
                deconv = tf.nn.conv2d_transpose(list_input[i], var, output_shape,
                                                strides=strides, padding='SAME', data_format=data_format)
                deconv.set_shape(new_shape)
                list_output.append(deconv)

    return list_output


def dropout(list_input, keep_prob):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            output = tf.nn.dropout(list_input[i], keep_prob=keep_prob)
            list_output.append(output)
    return list_output


def avg_pool(list_input, ksize, strides, padding, data_format):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            output = tf.nn.avg_pool(list_input[i], ksize, strides, padding, data_format)
            list_output.append(output)
    return list_output


def concat(list_input, axis):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input[0])):
        with tf.device('/gpu:%d' % i):
            to_concat = []
            for j in range(len(list_input)):
                to_concat.append(list_input[j][i])
            output = tf.concat(to_concat, axis)
            list_output.append(output)
    return list_output
