"""
mg means multiple gpus. The idea here is very simple: changing the inputs and outputs to a list whose length is equal
to the number of gpus. All operations are defined under `with tf.device('/gpu:%d' % i):`.
Except batch normalization, all functions have no difference from the implementation in a single gpu.

data_format = 'NHWC'
float_type = tf.float32
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

float_type = tf.float32


def stride_arr(stride):
    return [1, stride, stride, 1]


def input_data(list_images):
    """
    images are alwyas NHWC.
    """
    assert type(list_images) == list
    return list_images


def cast(list_inputs, new_float_type=None):
    if new_float_type is None:
        new_float_type = float_type

    list_output = []
    for i in range(len(list_inputs)):
        with tf.device('/gpu:%d' % i):
            list_output.append(tf.cast(list_inputs[i], new_float_type))
    return list_output


def resize_images(list_images, output_size, method='bilinear', train_conv2dt=False, stride_conv2dt=8):
    assert type(list_images) == list

    list_output = []
    if train_conv2dt:
        return conv2d_transpose('conv_transpose', list_images,
                                ksize=stride_conv2dt * 2,
                                stride=stride_conv2dt)

    for i in range(len(list_images)):
        with tf.device('/gpu:%d' % i):
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


def max_pool(list_input, ksize, stride):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            _x = tf.nn.max_pool(list_input[i],
                                ksize=stride_arr(ksize),
                                strides=stride_arr(stride),
                                padding='SAME')
            list_output.append(_x)
    return list_output


def global_avg_pool(list_input):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            _x = tf.reduce_mean(list_input[i], [1, 2], keep_dims=True)
            list_output.append(_x)
    return list_output


def conv2d_same(list_input, out_channels, kernel_size, stride,
                rate=1, initializer='he', he_init_std=None, scope=None):
    # ======================== Checking valid values =========================
    if initializer not in ['he', 'xavier']:
        raise TypeError(
            "Only two initializers are supported at this moment: 'he' or 'xavier', "
            "%s is an unknown initializer." % initializer)
    if not isinstance(stride, int):
        raise TypeError("Expecting an int for stride but %s is got." % stride)
    assert type(list_input) == list
    # ======================== Setting default values =========================
    in_channels = list_input[0].get_shape().as_list()[-1]

    if initializer == 'he':
        if he_init_std is None:
            n = kernel_size * kernel_size * out_channels
            std = np.sqrt(2.0 / n)
        else:
            std = he_init_std
        initializer = tf.random_normal_initializer(stddev=std)
    else:
        initializer = tf.contrib.layers.xavier_initializer()

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
                    dtype=float_type
                )
                if stride == 1:
                    if rate > 1:
                        _x = tf.nn.atrous_conv2d(list_input[i], kernel, rate, 'SAME')
                        list_output.append(_x)
                    else:
                        _x = tf.nn.conv2d(list_input[i], kernel, stride_arr(stride), 'SAME')
                        list_output.append(_x)
                else:
                    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
                    pad_total = kernel_size_effective - 1
                    pad_beg = pad_total // 2
                    pad_end = pad_total - pad_beg
                    inputs = tf.pad(list_input[i],
                                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
                    if rate > 1:
                        _x = tf.nn.atrous_conv2d(inputs, kernel, rate, 'SAME')
                        list_output.append(_x)
                    else:
                        _x = tf.nn.conv2d(inputs, kernel, stride_arr(stride), 'VALID')
                        list_output.append(_x)
    return list_output


def batch_norm(list_input, stats_mode,
               use_gamma=True, use_beta=True, bn_epsilon=1e-5, bn_ema=0.9, scope='BatchNorm'):
    def get_bn_variables(n_out, use_scale, use_bias):
        if use_bias:
            beta = tf.get_variable('beta', [n_out],
                                   initializer=tf.constant_initializer(), trainable=True, dtype=float_type)
        else:
            beta = tf.zeros([n_out], name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [n_out],
                                    initializer=tf.constant_initializer(1.0), trainable=True, dtype=float_type)
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
    assert type(list_input) == list

    # ======================== Setting default values =========================
    shape = list_input[0].get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]

    # ======================== Main operations =============================
    if 'gather' not in stats_mode:
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(scope, reuse=i > 0):
                    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_gamma, use_beta)
                    xn = tf.nn.batch_normalization(
                        list_input[i], moving_mean, moving_var, beta, gamma, bn_epsilon)
                    list_output.append(xn)
        return list_output

    if len(list_input) == 1:
        # use fused_bn
        list_output = []
        with tf.variable_scope(scope):
            beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_gamma, use_beta)
            xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
                list_input[0], gamma, beta, epsilon=bn_epsilon, is_training=True)
            xn = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, bn_ema)
            list_output.append(xn)
        return list_output

    means = []
    square_means = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            batch_mean = tf.reduce_mean(list_input[i], [0, 1, 2])
            batch_square_mean = tf.reduce_mean(tf.square(list_input[i]), [0, 1, 2])
            means.append(batch_mean)
            square_means.append(batch_square_mean)

    with tf.device('/gpu:0'):  # if your GPUs have NVLinks and you've install NCCL2, you can change `/cpu:0` to `/gpu:0`
        shape = tf.shape(list_input[0])
        num = shape[0] * shape[1] * shape[2] * len(list_input)
        batch_mean = tf.reduce_mean(means, axis=0)
        batch_var = tf.reduce_mean(square_means, axis=0) - tf.square(batch_mean)
        batch_var *= tf.cast(num, float_type) / tf.cast(num-1, float_type)  # unbiased variance

    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(scope, reuse=i > 0):
                beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_gamma, use_beta)

                xn = tf.nn.batch_normalization(
                    list_input[i], batch_mean, batch_var, beta, gamma, bn_epsilon)

                if i > 0:
                    list_output.append(xn)
                else:
                    # gathering stats in the main gpu device only.
                    xn = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, bn_ema)
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


def softmax(list_input):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            output = tf.nn.softmax(list_input[i])
            list_output.append(output)
    return list_output


def bottleneck_residual(list_input, out_channels, stride, rate=1, initializer='he',
                        bn_stat_mode='gather', bn_use_gamma=True, bn_use_beta=True, bn_epsilon=1e-5, bn_ema=0.9):
    """Bottleneck v1 residual unit with 3 sub layers."""

    # ======================== Checking valid values =========================
    if initializer not in ['he', 'xavier']:
        raise TypeError(
            "Only two initializers are supported at this moment: 'he' or 'xavier', "
            "%s is an unknown initializer." % initializer)
    if not isinstance(stride, int):
        raise TypeError("Expecting an int for stride but %s is got." % stride)
    assert type(list_input) == list
    # ======================== Setting default values =========================
    if stride > 1:
        rate = 1  # because of atrous_conv2d.

    in_channels = list_input[0].get_shape().as_list()[-1]

    # ======================== Main operations =============================
    orig_x = list_input
    with tf.variable_scope('bottleneck_v1/conv1'):
        """1x1, 64->64"""
        x = conv2d_same(list_input, out_channels // 4, 1, 1, 1, initializer)
        x = batch_norm(x, bn_stat_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
        x = relu(x)

    with tf.variable_scope('bottleneck_v1/conv2'):
        """3x3, 64->64"""
        x = conv2d_same(x, out_channels // 4, 3, stride, rate, initializer)
        x = batch_norm(x, bn_stat_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
        x = relu(x)

    with tf.variable_scope('bottleneck_v1/conv3'):
        """1x1, 64->256"""
        x = conv2d_same(x, out_channels, 1, 1, 1, initializer)
        x = batch_norm(x, bn_stat_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)

    """1x1, 64->256 or 256==256, do nothing."""
    if in_channels != out_channels:
        with tf.variable_scope('bottleneck_v1/shortcut'):
            orig_x = conv2d_same(orig_x, out_channels, 1, stride, initializer=initializer)
            orig_x = batch_norm(orig_x, bn_stat_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
    else:
        if stride > 1:
            orig_x = max_pool(orig_x, 1, 2)

    for i in range(len(x)):
        with tf.device('/gpu:%d' % i):
            x[i] += orig_x[i]

    x = relu(x)
    return x


def subsample(list_input, stride):
    if stride == 1:
        return list_input
    else:
        return max_pool(list_input, 1, stride)


def bottleneck_residual_v2(list_input, out_channels, stride, rate=1, initializer='he',
                           bn_stat_mode='train', bn_use_gamma=True, bn_use_beta=True, bn_epsilon=1e-5, bn_ema=0.9):
    """Bottleneck v2 residual unit with 3 sub layers."""

    # ======================== Checking valid values =========================
    if initializer not in ['he', 'xavier']:
        raise TypeError(
            "Only two initializers are supported at this moment: 'he' or 'xavier', "
            "%s is an unknown initializer." % initializer)
    if not isinstance(stride, int):
        raise TypeError("Expecting an int for stride but %s is got." % stride)
    assert list_input[0].get_shape().ndims == 4, 'inputs should have rank 4.'
    # ======================== Setting default values =========================
    in_channels = list_input[0].get_shape().as_list()[-1]

    # ======================== Main operations =============================
    with tf.variable_scope('bottleneck_v2'):
        preact = batch_norm(list_input, bn_stat_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
        preact = relu(preact)

        with tf.variable_scope('shortcut'):
            if out_channels == in_channels:
                shortcut = subsample(list_input, stride)
            else:
                shortcut = conv_bias_relu(preact, out_channels, 1, stride,
                                          relu=False,
                                          initializer=initializer, bias_scope='biases')

        with tf.variable_scope('conv1'):
            residual = conv2d_same(preact, out_channels // 4, 1, 1, 1, initializer)
            residual = batch_norm(residual, bn_stat_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
            residual = relu(residual)
        with tf.variable_scope('conv2'):
            residual = conv2d_same(residual, out_channels // 4, 3, stride, rate, initializer)
            residual = batch_norm(residual, bn_stat_mode, bn_use_gamma, bn_use_beta, bn_epsilon, bn_ema)
            residual = relu(residual)
        with tf.variable_scope('conv3'):
            residual = conv_bias_relu(residual, out_channels, 1, 1, relu=False, initializer=initializer,
                                      bias_scope='biases')

        for i in range(len(residual)):
            with tf.device('/gpu:%d' % i):
                residual[i] += shortcut[i]

    return residual


def fully_connected(list_input, out_channels, initializer='he'):
    """convolution 1x1 layer for final output."""
    list_x = conv2d_same(list_input, out_channels, 1, 1, initializer=initializer)

    list_output = []
    for i in range(len(list_x)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope('biases', reuse=(i > 0)):
                b = tf.get_variable('', [out_channels], initializer=tf.constant_initializer(0.01))
                output = tf.nn.bias_add(list_x[i], b)
                list_output.append(output)

    return list_output


def conv_bias_relu(list_input, out_channels, kernel_size, stride, relu=True,
                   initializer='he', bias_scope='biases'):
    list_x = conv2d_same(list_input, out_channels, kernel_size, stride, initializer=initializer)

    list_bias = []
    for i in range(len(list_x)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope(bias_scope, reuse=(i > 0)):
                b = tf.get_variable('', [out_channels], initializer=tf.constant_initializer(0.01))
                output = tf.nn.bias_add(list_x[i], b)
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
                     ksize=4, stride=2):
    # TODO: This does not perform the same as resize_image. check the difference.
    assert type(list_input) == list
    strides = [1, stride, stride, 1]
    list_output = []
    in_features = list_input[0].get_shape()[3].value

    if out_channels is None:
        out_channels = in_features

    # Compute shape out of Bottom
    in_shape = list_input[0].get_shape()

    h = in_shape[1] * stride
    w = in_shape[2] * stride
    new_shape = [in_shape[0], h, w, out_channels]

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
                                      shape=weights.shape)
                deconv = tf.nn.conv2d_transpose(list_input[i], var, output_shape,
                                                strides=strides, padding='SAME')
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


def avg_pool(list_input, ksize, strides, padding):
    assert type(list_input) == list
    list_output = []
    for i in range(len(list_input)):
        with tf.device('/gpu:%d' % i):
            output = tf.nn.avg_pool(list_input[i], ksize, strides, padding)
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
