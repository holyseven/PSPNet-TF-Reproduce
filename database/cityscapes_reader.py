import numpy as np
import tensorflow as tf
import math

IGNORE_LABEL = 255
IMG_MEAN = np.array((72.41519599, 82.93553322, 73.18188461), dtype=np.float32)  # RGB


def image_scaling(img, label, scale_rate):
    if scale_rate is None:
        print 'applying random scale [0.5, 2]...'
        scale = tf.random_uniform([1], minval=0.5, maxval=2.0, seed=None)
    else:
        print 'applying random scale [%f, %f]...' % (scale_rate[0], scale_rate[1])
        scale = tf.random_uniform([1], minval=scale_rate[0], maxval=scale_rate[1], seed=None)
    h_new = tf.to_int32(tf.multiply(tf.cast(tf.shape(img)[0], tf.float32), scale))
    w_new = tf.to_int32(tf.multiply(tf.cast(tf.shape(img)[1], tf.float32), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

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
                            lambda: (crop_h - image_shape[0])/2,
                            lambda: tf.constant(0))
    offset_width = tf.cond(tf.less(image_shape[1], crop_w),
                           lambda: (crop_w - image_shape[1])/2,
                           lambda: tf.constant(0))
    combined_pad = tf.image.pad_to_bounding_box(combined, offset_height=offset_height, offset_width=offset_width,
                                                target_height=tf.maximum(crop_h, image_shape[0]),
                                                target_width=tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    # label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop


def _read_labeled_image_list(data_dir, data_sub):
    if data_sub not in ['train', 'val', 'test', 'train_extra']:
        print 'data sub should be train, val, test or train_extra'
        return
    import glob
    images_filename_proto = data_dir + '/leftImg8bit/' + data_sub + '/*/*.png'
    images = sorted(glob.glob(images_filename_proto))

    labels_filename_proto = data_dir + '/gt/' + data_sub + '/*/*.png'
    labels = sorted(glob.glob(labels_filename_proto))

    # for just checking they are corresponded.
    for i in range(len(images)):
        if images[i].split('leftImg8bit')[1] == labels[i].split('gt')[1]:
            continue

        print i, images[i], labels[i]

    return images, labels


def read_labeled_image_list(data_dir, data_sub):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_sub: path to the file with lines of the form '/path/to/image /path/to/mask'. 'train' or 'val'

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    if type(data_sub) is list:
        # use more for training.
        images, labels = [], []
        for each_data_set in data_sub:
            each_image_set, each_label_set = _read_labeled_image_list(data_dir, each_data_set)
            images += each_image_set
            labels += each_label_set
        return images, labels
    else:
        return _read_labeled_image_list(data_dir, data_sub)


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

        image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
                                                             background_color, validate_indices=False))

    image_rotated = tf.transpose(tf.stack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated


def generate_crops_for_training(input_queue, input_size, random_scale, random_mirror, random_blur,
                                random_rotate, color_switch, scale_rate):
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_png(img_contents, channels=3)  # r,g,b
    img = tf.cast(img, dtype=tf.float32)

    if random_blur:
        img = tf.image.random_brightness(img, max_delta=63. / 255.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.0)

    # Extract mean.
    img -= IMG_MEAN

    if color_switch:
        # this depends on the model we are using.
        # if provided by tensorflow, no need to switch color
        # if a model converted from caffe, need to switch color.
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.concat([img_b, img_g, img_r], 2)

    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size

        # Randomly mirror the images and labels.
        if random_mirror:
            print 'applying random mirror ...'
            img, label = image_mirroring(img, label)

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label, scale_rate)

        # Randomly rotate the images and lables.
        if random_rotate:
            print 'applying random rotation...'
            rd_rotatoin = tf.random_uniform([], -10.0, 10.0)
            angle = rd_rotatoin / 180 * math.pi
            img = rotate_image_tensor(img, angle, mode='black')
            label = rotate_image_tensor(label, angle, mode='white')

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, IGNORE_LABEL)

    return img, label


def output_one_image(image_addr, label_addr, color_switch):
    """

    :param image_addr:
    :param label_addr:
    :param color_switch:
    :return: image and label with only soustracting the MEAN.
    """
    img_contents = tf.read_file(image_addr)

    img = tf.image.decode_png(img_contents, channels=3)  # r,g,b
    img = tf.cast(img, dtype=tf.float32)

    # Extract mean.
    img -= IMG_MEAN

    if color_switch:
        # this depends on the model we are using.
        # if provided by tensorflow, no need to switch color
        # if a model converted from caffe, need to switch color.
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.concat([img_b, img_g, img_r], 2)

    if label_addr is None:
        label = img
    else:
        label_contents = tf.read_file(label_addr)
        label = tf.image.decode_png(label_contents, channels=1)

    return tf.expand_dims(img, dim=0), tf.expand_dims(label, dim=0)


def output_test_set(server, color_switch):
    if server == 1:
        data_dir = '/c3d/d1/cityscape/'
    elif server == 2:
        data_dir = '/home/lab/lixuhong/data/cityscapes/'
    else:
        data_dir = '/home/jacques/workspace/database/cs'

    import glob
    images_filename_proto = data_dir + '/leftImg8bit/test' + '/*/*.png'
    images = sorted(glob.glob(images_filename_proto))
    queue = tf.train.slice_input_producer([images], shuffle=False)
    print 'Database has %d images.' % len(images)
    img_contents = tf.read_file(queue[0])

    img = tf.image.decode_png(img_contents, channels=3)  # r,g,b
    img = tf.cast(img, dtype=tf.float32)

    # Extract mean.
    img -= IMG_MEAN

    if color_switch:
        # this depends on the model we are using.
        # if provided by tensorflow, no need to switch color
        # if a model converted from caffe, need to switch color.
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
        img = tf.concat([img_b, img_g, img_r], 2)

    return tf.expand_dims(img, dim=0), queue


class CityScapesReader(object):
    """Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    """

    def __init__(self, server, data_list, input_size, random_scale,
                 random_mirror, random_blur, random_rotate, color_switch, scale_rate=None):

        if server == 1:
            data_dir = '/c3d/d1/cityscape/'
        elif server == 2:
            data_dir = '/home/lab/lixuhong/data/cityscapes/'
        else:
            data_dir = '/home/jacques/workspace/database/cs'

        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)
        print 'Database has %d images.' % len(self.image_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        shuffle = ('train' == data_list) or 'train' in data_list
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=shuffle, capacity=128)
        self.image, self.label = generate_crops_for_training(self.queue, self.input_size,
                                                             random_scale, random_mirror, random_blur, random_rotate,
                                                             color_switch, scale_rate=scale_rate)

    def dequeue(self, batch_size):
        """Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks."""
        image_batch, label_batch = tf.train.batch([self.image, self.label],
                                                  batch_size, batch_size * 4, 128 * batch_size)

        return image_batch, tf.cast(label_batch, dtype=tf.int32)

    def dequeue_without_crops(self, batch_size, height=1024, width=2048):
        img_contents = tf.read_file(self.queue[0])
        img = tf.image.decode_png(img_contents, channels=3)  # r,g,b
        img = tf.cast(img, dtype=tf.float32)
        img -= IMG_MEAN
        # [h, w, 3]

        label_contents = tf.read_file(self.queue[1])
        label = tf.image.decode_png(label_contents, channels=1)
        # [h, w, 1]

        img.set_shape((height, width, 3))
        label.set_shape((height, width, 1))

        image_batch, label_batch = tf.train.batch([img, label],
                                                  batch_size, batch_size * 2, 4 * batch_size)
        return image_batch, tf.cast(label_batch, dtype=tf.int32)

