import tensorflow as tf
import numpy as np
import math
from database import helper
from tensorflow.contrib.image import rotate
import glob


def find_data_path(dataset):
    if dataset == 'Cityscapes':
        img_mean = np.array((72.41519599, 82.93553322, 73.18188461), dtype=np.float32)  # RGB, Cityscapes.
        num_classes = 19
        data_dir = './database/cityscapes'
    elif dataset == 'SBD':
        img_mean = np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32)  # RGB, SBD/Pascal VOC.
        num_classes = 21
        data_dir = './database/SBD_all'
    elif 'ADE' in dataset:
        img_mean = np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32)  # RGB, SBD/Pascal VOC.
        num_classes = 150
        data_dir = './database/ADEChallengeData2016'
    else:
        raise ValueError("Unknown database %s" % dataset)

    return data_dir, img_mean, num_classes


def _read_cityscapes_image_label_list(data_dir, data_sub):
    if data_sub not in ['train', 'val', 'test', 'train_extra']:
        print('data sub should be train, val, test or train_extra')
        return
    import glob
    images_filename_proto = data_dir + '/leftImg8bit/' + data_sub + '/*/*.png'
    images = sorted(glob.glob(images_filename_proto))

    labels_filename_proto = data_dir + '/gt/' + data_sub + '/*/*.png'
    labels = sorted(glob.glob(labels_filename_proto))

    assert len(images) == len(labels), 'images and labels have different numbers of examples. ' \
                                       'Suggestion: add more constraint on the filename_proto, ' \
                                       'or move undesired images to other directory.'
    # TODO: verify if incorrectly read labels containing labelIds [0, 34].
    # For now, see the prepation at
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdInstanceImgs.py

    # for just checking they are corresponded.
    for i in range(len(images)):
        if images[i].split('leftImg8bit')[1] == labels[i].split('gt')[1]:
            continue

        print('< Error >', i, images[i], labels[i])

    return images, labels


def _read_sbd_image_label_list(data_dir, data_sub):
    if data_sub not in ['train', 'val', 'test']:
        print('data sub should be train, val or test')
        return
    f = open(data_dir + '/' + data_sub + '.txt', 'r')
    images = []
    lables = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError:  # for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        lables.append(data_dir + mask)
    return images, lables


def _read_ade20k_image_label_list(data_dir, data_sub):
    if data_sub not in ['train', 'val']:
        print('data sub should be train or val')
        return

    if data_sub == 'train':
        data_sub = 'training'
    else:
        data_sub = 'validation'

    images_filename_proto = data_dir + '/images/' + data_sub + '/*.jpg'
    images = sorted(glob.glob(images_filename_proto))

    labels_filename_proto = data_dir + '/annotations/' + data_sub + '/*.png'
    labels = sorted(glob.glob(labels_filename_proto))

    assert len(images) == len(labels)

    # for just checking they are corresponded.
    for i in range(len(images)):
        if images[i].split('.jpg')[0].split('/')[-1] == labels[i].split('.png')[0].split('/')[-1]:
            continue

        print('< Error >', i, images[i], labels[i])

    return images, labels


def read_labeled_image_list(dataset, data_dir, data_sub):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_sub: path to the file with lines of the form '/path/to/image /path/to/mask'. 'train' or 'val'

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    if dataset == 'SBD':
        path_read_func = _read_sbd_image_label_list
    elif dataset == 'Cityscapes':
        path_read_func = _read_cityscapes_image_label_list
    elif 'ADE' in dataset:
        path_read_func = _read_ade20k_image_label_list
    else:
        raise ValueError("Unknown database %s" % dataset)

    if type(data_sub) is list:
        # use more for training.
        images, labels = [], []
        for each_data_set in data_sub:
            each_image_set, each_label_set = path_read_func(data_dir, each_data_set)
            images += each_image_set
            labels += each_label_set
        return images, labels
    else:
        return path_read_func(data_dir, data_sub)


class QueueBasedImageReader(object):
    """Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    """

    def __init__(self, dataset, data_list):
        self.dataset_name = dataset
        self.data_dir, self.img_mean, self.num_classes = find_data_path(dataset)
        self.data_list = data_list
        self.image_list, self.label_list = read_labeled_image_list(dataset, self.data_dir, self.data_list)
        assert len(self.image_list) > 0, 'No images are found.'
        print('Database has %d images.' % len(self.image_list))
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        shuffle = ('train' == self.data_list) or 'train' in self.data_list
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=shuffle, capacity=128)

    def get_batch(self, batch_size, crop_size, random_mirror, random_blur, random_rotate,
                  color_switch, random_scale, scale_rate=None):
        label_contents = tf.read_file(self.queue[1])
        label = tf.image.decode_png(label_contents, channels=1)
        if 'ADE' in self.dataset_name:  # the first label (0) of ADE is background.
            label -= 1

        img_contents = tf.read_file(self.queue[0])
        img = tf.image.decode_image(img_contents, channels=3)
        img.set_shape((None, None, 3))  # decode_image does not returns no shape.
        img = tf.cast(img, dtype=tf.float32)

        if random_blur:
            img = tf.image.random_brightness(img, max_delta=63. / 255.)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

        # Extract mean.
        img -= self.img_mean

        if color_switch:
            # this depends on the model we are using.
            # if provided by tensorflow, no need to switch color
            # if a model converted from caffe, need to switch color.
            img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
            img = tf.concat([img_b, img_g, img_r], 2)

        # Randomly mirror the images and labels.
        if random_mirror:
            print('\t applying random mirror ...')
            img, label = helper.image_mirroring(img, label)

        # Randomly scale the images and labels.
        if random_scale:
            if scale_rate is not None:
                print('\t applying random scale [%f, %f]...' % (scale_rate[0], scale_rate[1]))
            else:
                print('\t applying random scale [0.5, 2.0]...')
            img, label = helper.image_scaling(img, label, scale_rate)

        # Randomly rotate the images and lables.
        if random_rotate:
            print('\t applying random rotation...')
            rd_rotatoin = tf.random_uniform([], -10.0, 10.0)
            angle = rd_rotatoin / 180.0 * math.pi

            img = rotate(img, angle, 'BILINEAR')
            label -= 255
            label = rotate(label, angle, 'NEAREST')
            label += 255

        # Randomly crops the images and labels.
        img, label = helper.random_crop_and_pad_image_and_labels(img, label, crop_size, crop_size)

        image_batch, label_batch = tf.train.batch([img, label],
                                                  batch_size, batch_size * 4, 32 * batch_size)

        return image_batch, tf.cast(label_batch, dtype=tf.int32)

    def get_eval_batch(self, color_switch):
        label_contents = tf.read_file(self.queue[1])
        label = tf.image.decode_png(label_contents, channels=1)
        if 'ADE' in self.dataset_name:  # the first label (0) of ADE is background.
            label -= 1

        img_contents = tf.read_file(self.queue[0])
        img = tf.image.decode_image(img_contents, channels=3)
        img.set_shape((None, None, 3))  # decode_image does not returns no shape.
        img = tf.cast(img, dtype=tf.float32)
        # Extract mean.
        img -= self.img_mean
        if color_switch:
            img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
            img = tf.concat([img_b, img_g, img_r], 2)

        return img, label, self.queue[0]


class ImageReader(object):
    """
    for segmentation.
    """
    def __init__(self, dataset, data_list):
        self.dataset_name = dataset
        self.data_dir, self.img_mean, self.num_classes = find_data_path(dataset)
        self.data_list = data_list  # train, val (eval), or test.
        self.image_list, self.label_list = read_labeled_image_list(dataset, self.data_dir, self.data_list)
        assert len(self.image_list) > 0, 'No images are found.'
        assert len(self.image_list) == len(self.label_list)
        print('[info] Database %s has %d images in %s.' % (self.dataset_name, len(self.image_list), self.data_list))

    def get_batch_iterator(self, batch_size, crop_size, random_mirror, random_blur, random_rotate,
                           color_switch, random_scale, scale_rate=None):
        """
        for training.
        """
        img_mean = self.img_mean
        dataset_name = self.dataset_name

        def _training_data_preprocess(image_filename, label_filename):
            img_contents = tf.read_file(image_filename)
            label_contents = tf.read_file(label_filename)
            img = tf.image.decode_image(img_contents, channels=3)
            img.set_shape((None, None, 3))  # decode_image does not returns no shape.
            img = tf.cast(img, dtype=tf.float32)

            if random_blur:
                img = tf.image.random_brightness(img, max_delta=63. / 255.)
                img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
                img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

            img -= img_mean

            if color_switch:
                img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
                img = tf.concat([img_b, img_g, img_r], 2)

            label = tf.image.decode_png(label_contents, channels=1)
            label = tf.cast(label, tf.int32)

            if 'ADE' in dataset_name:  # the first label (0) of ADE is background.
                label -= 1

            # Randomly mirror the images and labels.
            if random_mirror:
                print('\t applying random mirror ...')
                img, label = helper.image_mirroring(img, label)

            # Randomly scale the images and labels.
            if random_scale:
                if scale_rate is not None:
                    print('\t applying random scale [%f, %f]...' % (scale_rate[0], scale_rate[1]))
                else:
                    print('\t applying random scale [0.5, 2.0]...')
                img, label = helper.image_scaling(img, label, scale_rate)

            # Randomly rotate the images and lables.
            if random_rotate:
                print('\t applying random rotation...')
                rd_rotatoin = tf.random_uniform([], -10.0, 10.0)
                angle = rd_rotatoin / 180.0 * math.pi

                img = rotate(img, angle, 'BILINEAR')
                label -= 255
                label = rotate(label, angle, 'NEAREST')
                label += 255

            # Randomly crops the images and labels.
            img, label = helper.random_crop_and_pad_image_and_labels(img, label, crop_size, crop_size)

            return img, label

        images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        # use shard, but how?
        dataset = dataset.repeat()\
            .shuffle(batch_size * 100)\
            .map(_training_data_preprocess, num_parallel_calls=batch_size)

        batched_dataset = dataset.batch(batch_size)
        batched_dataset = batched_dataset.prefetch(1)
        iterator = batched_dataset.make_initializable_iterator()

        return iterator

    def get_eval_iterator(self, color_switch):
        """
        for eval and test. no scale.
        """
        img_mean = self.img_mean
        dataset_name = self.dataset_name

        def _eval_data_preprocess(image_filename, label_filename):
            img_contents = tf.read_file(image_filename)
            label_contents = tf.read_file(label_filename)

            img = tf.image.decode_image(img_contents, channels=3)
            img.set_shape((None, None, 3))  # decode_image does not returns no shape.
            img = tf.cast(img, dtype=tf.float32)

            img -= img_mean

            if color_switch:
                img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
                img = tf.concat([img_b, img_g, img_r], 2)

            label = tf.image.decode_png(label_contents, channels=1)

            if 'ADE' in dataset_name:  # the first label (0) of ADE is background.
                label -= 1

            return img, label, image_filename

        images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(_eval_data_preprocess, num_parallel_calls=4)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()

        return iterator

    def get_next_image(self):
        return

