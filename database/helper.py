import numpy as np
import tensorflow as tf


def rotate_image_tensor(image, angle, mode='black'):
    """
    Rotates a 3D tensor (HWD), which represents an image by given radian angle.

    New image has the same size as the input image.

    mode controls what happens to border pixels.
    mode = 'black' results in black bars (value 0 in unknown areas)
    mode = 'white' results in value 255 in unknown areas
    mode = 'ones' results in value 1 in unknown areas
    mode = 'repeat' keeps repeating the closest pixel known
    """

    s = tf.shape(image)
    assert s.get_shape()[0] == 3, "Input needs to be 3D."
    assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
    image_center = [tf.floor(tf.cast(s[0]/2, tf.float32)), tf.floor(tf.cast(s[1]/2, tf.float32))]

    # Coordinates of new image
    coord1 = tf.range(s[0])
    coord2 = tf.range(s[1])

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [s[1]])

    coord2_vec_unordered = tf.tile(coord2, [s[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - tf.to_int32(image_center[0])
    coord2_vec_centered = coord2_vec - tf.to_int32(image_center[1])

    coord_new_centered = tf.cast(tf.stack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.dynamic_stitch([0, 1, 2, 3], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find nearest neighbor in old image
    coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
    coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

    # Clip values to stay inside image coordinates
    if mode == 'repeat':
        coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0]-1)
        coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1]-1)
    else:
        outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0]-1), tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1]-1), tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord_old1_clipped = tf.boolean_mask(coord1_old_nn, tf.logical_not(outside_ind))
        coord_old2_clipped = tf.boolean_mask(coord2_old_nn, tf.logical_not(outside_ind))

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

    coord_old_clipped = tf.cast(tf.transpose(tf.stack([coord_old1_clipped, coord_old2_clipped]), [1, 0]), tf.int32)

    # Coordinates of the new image
    coord_new = tf.transpose(tf.cast(tf.stack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

    num_channels = image.get_shape().as_list()[2]
    image_channel_list = tf.split(image, num_channels, axis=2)

    image_rotated_channel_list = list()
    for image_channel in image_channel_list:
        image_chan_new_values = tf.gather_nd(tf.squeeze(image_channel), coord_old_clipped)

        if (mode == 'black') or (mode == 'repeat'):
            background_color = 0
        elif mode == 'ones':
            background_color = 1
        elif mode == 'white':
            background_color = 255
        else:
            background_color = 0

        image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
                                                             background_color, validate_indices=False))

    image_rotated = tf.transpose(tf.stack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated


def image_scaling(img, label, scale_rate):
    if scale_rate is None:
        scale = tf.random_uniform([1], minval=0.5, maxval=2.0, seed=None)
    else:
        scale = tf.random_uniform([1], minval=scale_rate[0], maxval=scale_rate[1], seed=None)
    h_new = tf.to_int32(tf.multiply(tf.cast(tf.shape(img)[0], tf.float32), scale))
    w_new = tf.to_int32(tf.multiply(tf.cast(tf.shape(img)[1], tf.float32), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, axis=[0])

    return img, label


def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    img = tf.cond(mirror_cond,
                  lambda: tf.reverse(img, [1]),
                  lambda: img)
    label = tf.cond(mirror_cond,
                    lambda: tf.reverse(label, [1]),
                    lambda: label)

    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat([image, label], 2)
    image_shape = tf.shape(image)
    offset_height = tf.cond(tf.less(image_shape[0], crop_h),
                            lambda: (crop_h - image_shape[0])//2,
                            lambda: tf.constant(0))
    offset_width = tf.cond(tf.less(image_shape[1], crop_w),
                           lambda: (crop_w - image_shape[1])//2,
                           lambda: tf.constant(0))

    # padding zeros.
    combined_pad = tf.image.pad_to_bounding_box(combined, offset_height=offset_height, offset_width=offset_width,
                                                target_height=tf.maximum(crop_h, image_shape[0]),
                                                target_width=tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, last_image_dim+1])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    # label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop


def decide_intersection(total_length, crop_length):
    stride = crop_length * 2 // 3
    times = (total_length - crop_length) // stride + 1
    cropped_starting = []
    for i in range(times):
        cropped_starting.append(stride*i)
    if total_length - cropped_starting[-1] > crop_length:
        cropped_starting.append(total_length - crop_length)
    return cropped_starting


def numpy_pad_image(image, total_padding_h, total_padding_w, image_padding_value=0):
    # (height, width, channel)
    assert len(image.shape) == 3
    pad_before_w = pad_after_w = 0
    pad_before_h = pad_after_h = 0
    if total_padding_h < 0:
        if total_padding_h % 2 == 0:
            pad_before_h = pad_after_h = - total_padding_h // 2
        else:
            pad_before_h = - total_padding_h // 2
            pad_after_h = - total_padding_h // 2 + 1
    if total_padding_w < 0:
        if total_padding_w % 2 == 0:
            pad_before_w = pad_after_w = - total_padding_w // 2
        else:
            pad_before_w = - total_padding_w // 2
            pad_after_w = - total_padding_w // 2 + 1
    image_crop = np.pad(image,
                        ((pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0, 0)),
                        mode='constant', constant_values=image_padding_value)
    return image_crop


def numpy_crop_image(image, dif_height, dif_width):
    # (height, width, channel)
    assert len(image.shape) == 3
    if dif_height < 0:
        pad_before_h = - dif_height // 2
        pad_after_h = dif_height // 2
        image = image[pad_before_h:pad_after_h]

    if dif_width < 0:
        pad_before_w = - dif_width // 2
        pad_after_w = dif_width // 2
        image = image[:, pad_before_w:pad_after_w]

    return image


def compute_confusion_matrix(label, prediction, confusion_matrix):
    """
    confusion_matrix inplace update.
    :param label:
    :param prediction:
    :param confusion_matrix:
    :return:
    """
    num_classes = confusion_matrix.shape[0]
    label = np.reshape(label, (-1))
    prediction = np.reshape(prediction, (-1))
    indice = np.where(label < num_classes)
    label = label[indice]
    prediction = prediction[indice]

    np.add.at(confusion_matrix, (label, prediction), 1)
    return confusion_matrix


def compute_iou(confusion_matrix):
    cm = confusion_matrix.astype(np.float)
    sum_over_row = np.sum(cm, 0)
    sum_over_col = np.sum(cm, 1)
    cm_diag = np.diag(cm)
    denominator = sum_over_row + sum_over_col - cm_diag

    iou = np.divide(cm_diag, denominator)
    print('[IoUs]: ', end='')
    for i in range(confusion_matrix.shape[0]):
        print('%.2f' % (iou[i]*100), '& ', end='')
    print('\n[mIoU]: %.2f' % (np.mean(iou)*100), '\\\\')
    return np.mean(iou)


def compute_iou_each_class(confusion_matrix):
    cm = confusion_matrix.astype(np.float)
    sum_over_row = np.sum(cm, 0)
    sum_over_col = np.sum(cm, 1)
    cm_diag = np.diag(cm)
    denominator = sum_over_row + sum_over_col - cm_diag
    return np.divide(cm_diag, denominator)


if __name__ == '__main__':
    print(decide_intersection(int(2048 * 1.75), 864))
    assert decide_intersection(int(2048 * 1.75), 864) == [0, 576, 1152, 1728, 2304, 2720]
    assert decide_intersection(int(1024 * 1.75), 864) == [0, 576, 928]
    assert decide_intersection(int(2048 * 0.5), 864) == [0, 160]
    # print decide_intersection(int(1024 * 0.5), 864)
    assert decide_intersection(864, 864) == [0]
    assert decide_intersection(2048, 720) == [0, 480, 960, 1328]

    assert decide_intersection(875, 480) == [0, 320, 395]
    assert decide_intersection(800, 480) == [0, 320]
    assert decide_intersection(int(799.75), 480) == [0, 319]

    # r_img = splitter.reassemble_crops(splitter.get_split_crops())


