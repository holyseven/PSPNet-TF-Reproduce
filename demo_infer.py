import tensorflow as tf
import os
import numpy as np
import cv2

from args import FLAGS
from database import reader, helper, helper_cityscapes
from model import pspnet_mg
from experiment_manager.utils import sorted_str_dict


def gpu_num():
    return len(FLAGS.visible_gpus.split(','))


def infer(image_filename, i_ckpt):

    # < single gpu version >
    # < use FLAGS.batch_size as batch size, it is a number of crops running each time >
    # < use FLAGS.weight_ckpt as i_ckpt >
    # < use FLAGS.database to indicate img_mean and num_classes >

    with tf.device('/cpu:0'):
        _, img_mean, num_classes = reader.find_data_path(FLAGS.database)
        img_contents = tf.read_file(image_filename)
        img = tf.image.decode_image(img_contents, channels=3)
        img.set_shape((None, None, 3))  # decode_image does not returns no shape.
        img = tf.cast(img, dtype=tf.float32)
        img -= img_mean

    crop_size = FLAGS.test_image_size
    # < network >
    model = pspnet_mg.PSPNetMG(num_classes, FLAGS.network, gpu_num(), three_convs_beginning=FLAGS.three_convs_beginning)
    images_pl = [tf.placeholder(tf.float32, [None, crop_size, crop_size, 3])]
    eval_probas_op = model.build_forward_ops(images_pl)

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init)

    loader = tf.train.Saver(max_to_keep=0)
    loader.restore(sess, i_ckpt)

    scales = [1.0]
    if FLAGS.ms == 1:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    def run_once(input_image):
        H, W, channel = input_image.shape

        # < in case that input_image is smaller than crop_size >
        dif_height = H - crop_size
        dif_width = W - crop_size
        if dif_height < 0 or dif_width < 0:
            input_image = helper.numpy_pad_image(input_image, dif_height, dif_width)
            H, W, channel = input_image.shape

        # < split this image into crops >
        split_crops = []
        heights = helper.decide_intersection(H, crop_size)
        widths = helper.decide_intersection(W, crop_size)
        for height in heights:
            for width in widths:
                image_crop = input_image[height:height + crop_size, width:width + crop_size]
                split_crops.append(image_crop[np.newaxis, :])

        # < >
        num_chunks = int((len(split_crops) - 1) / FLAGS.batch_size) + 1
        proba_crops_list = []
        for chunk_i in range(num_chunks):
            feed_dict = {}
            start = chunk_i * FLAGS.batch_size
            end = min((chunk_i+1)*FLAGS.batch_size, len(split_crops))
            feed_dict[images_pl[0]] = np.concatenate(split_crops[start:end])
            proba_crops_part = sess.run(eval_probas_op, feed_dict=feed_dict)
            proba_crops_list.append(proba_crops_part[0])

        proba_crops = np.concatenate(proba_crops_list)

        # < reassemble >
        reassemble = np.zeros((H, W, num_classes), np.float32)
        index = 0
        for height in heights:
            for width in widths:
                reassemble[height:height + crop_size, width:width + crop_size] += proba_crops[index]
                index += 1

        # < crop to original image >
        if dif_height < 0 or dif_width < 0:
            reassemble = helper.numpy_crop_image(reassemble, dif_height, dif_width)

        return reassemble

    orig_one_image = sess.run(img)
    orig_height, orig_width, channel = orig_one_image.shape
    total_proba = np.zeros((orig_height, orig_width, num_classes), dtype=np.float32)
    for scale in scales:
        if scale != 1.0:
            one_image = cv2.resize(orig_one_image, dsize=(0, 0), fx=scale, fy=scale)
        else:
            one_image = np.copy(orig_one_image)

        proba = run_once(one_image)
        if FLAGS.mirror == 1:
            proba_mirror = run_once(one_image[:, ::-1])
            proba += proba_mirror[:, ::-1]

        if scale != 1.0:
            proba = cv2.resize(proba, (orig_width, orig_height))

        total_proba += proba

    prediction = np.argmax(total_proba, axis=-1)
    cv2.imwrite('./demo_examples/demo_prediction.png', prediction)
    if FLAGS.database == 'Cityscapes':
        cv2.imwrite('./demo_examples/demo_color.png',
                    cv2.cvtColor(helper_cityscapes.coloring(prediction), cv2.COLOR_BGR2RGB))

    return prediction


def main(_):
    print(sorted_str_dict(FLAGS.__dict__))
    assert gpu_num() == 1, 'it is a single-GPU version because multiple GPUs are not helpful.'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpus

    infer('./demo_examples/berlin_000000_000019_leftImg8bit.png', FLAGS.weights_ckpt)


if __name__ == '__main__':
    tf.app.run()
