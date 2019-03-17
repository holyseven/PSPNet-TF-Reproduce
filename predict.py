import tensorflow as tf
import os
import datetime
import numpy as np
import cv2

from args import FLAGS
from database import reader, helper, helper_cityscapes
from model import pspnet_mg
from experiment_manager.utils import sorted_str_dict


def gpu_num():
    return len(FLAGS.visible_gpus.split(','))


def prediction_image_create(image_filename, key_word='leftImg8bit'):
    return str(image_filename).split('/')[-1].split(key_word)[0] + 'prediction.png'


def coloring_image_create(image_filename):
    return str(image_filename).split('/')[-1].split('leftImg8bit')[0] + 'coloring.png'


def predict(i_ckpt):

    # < single gpu version >
    # < use FLAGS.batch_size as batch size >
    # < use FLAGS.weight_ckpt as i_ckpt >

    reader_init = []
    with tf.device('/cpu:0'):
        if FLAGS.reader_method == 'queue':
            eval_image_reader = reader.QueueBasedImageReader(FLAGS.database, FLAGS.test_subset)
            eval_image, eval_label, eval_image_filename = eval_image_reader.get_eval_batch(FLAGS.color_switch)
        else:
            eval_image_reader = reader.ImageReader(FLAGS.database, FLAGS.test_subset)
            eval_reader_iterator = eval_image_reader.get_eval_iterator(FLAGS.color_switch)
            eval_image, eval_label, eval_image_filename = eval_reader_iterator.get_next()  # one image.
            reader_init.append(eval_reader_iterator.initializer)

    crop_size = FLAGS.test_image_size
    # < network >
    model = pspnet_mg.PSPNetMG(eval_image_reader.num_classes, FLAGS.network, gpu_num(), FLAGS.initializer,
                               FLAGS.weight_decay_mode, FLAGS.fine_tune_filename, FLAGS.optimizer, FLAGS.momentum,
                               FLAGS.train_like_in_caffe, FLAGS.three_convs_beginning, FLAGS.new_layer_names,
                               consider_dilated=FLAGS.consider_dilated)
    images_pl = [tf.placeholder(tf.float32, [None, crop_size, crop_size, 3])]
    eval_probas_op = model.build_forward_ops(images_pl)

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)

    if FLAGS.reader_method == 'queue':
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    init = [tf.global_variables_initializer(), tf.local_variables_initializer()] + reader_init
    sess.run(init)

    loader = tf.train.Saver(max_to_keep=0)
    loader.restore(sess, i_ckpt)

    prefix = i_ckpt.split('model.ckpt')[0] + FLAGS.test_subset + '_set/'
    if not os.path.exists(prefix) and 'test' in FLAGS.test_subset:
        os.mkdir(prefix)
        print('saving predictions to', prefix)

    confusion_matrix = np.zeros((eval_image_reader.num_classes, eval_image_reader.num_classes), dtype=np.int64)
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
        reassemble = np.zeros((H, W, eval_image_reader.num_classes), np.float32)
        index = 0
        for height in heights:
            for width in widths:
                reassemble[height:height + crop_size, width:width + crop_size] += proba_crops[index]
                index += 1

        # < crop to original image >
        if dif_height < 0 or dif_width < 0:
            reassemble = helper.numpy_crop_image(reassemble, dif_height, dif_width)

        return reassemble

    for i in range(len(eval_image_reader.image_list)):
        orig_one_image, one_label, image_filename = sess.run([eval_image, eval_label, eval_image_filename])
        orig_height, orig_width, channel = orig_one_image.shape
        total_proba = np.zeros((orig_height, orig_width, eval_image_reader.num_classes), dtype=np.float32)
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
        helper.compute_confusion_matrix(one_label, prediction, confusion_matrix)

        if 'test' in FLAGS.test_subset:
            if FLAGS.database == 'Cityscapes':
                cv2.imwrite(prefix + prediction_image_create(image_filename),
                            helper_cityscapes.trainid_to_labelid(prediction))
                if FLAGS.coloring == 1:
                    cv2.imwrite(prefix + coloring_image_create(image_filename),
                                cv2.cvtColor(helper_cityscapes.coloring(prediction), cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite(prefix + prediction_image_create(image_filename, key_word='.'), prediction)

        if i % 100 == 0:
            print('%s %s] %d / %d. iou updating' \
                  % (str(datetime.datetime.now()), str(os.getpid()), i, len(eval_image_reader.image_list)))
            helper.compute_iou(confusion_matrix)

    print('%s %s] %d / %d. iou updating' \
          % (str(datetime.datetime.now()), str(os.getpid()),
             len(eval_image_reader.image_list),
             len(eval_image_reader.image_list)))
    miou = helper.compute_iou(confusion_matrix)

    log_file = i_ckpt.split('model.ckpt')[0] + 'predict-ms' + str(FLAGS.ms) + '-mirror' + str(FLAGS.mirror) + '.txt'
    f_log = open(log_file, 'w')
    f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')
    ious = helper.compute_iou_each_class(confusion_matrix)
    f_log.write(str(ious) + '\n')
    for i in range(confusion_matrix.shape[0]):
        f_log.write(str(ious[i]) + '\n')
    f_log.write(str(miou) + '\n')

    if FLAGS.reader_method == 'queue':
        coord.request_stop()
        coord.join(threads)

    return


def main(_):
    print(sorted_str_dict(FLAGS.__dict__))
    assert gpu_num() == 1, 'it is a single-GPU version because multiple GPUs are not helpful.'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpus

    predict(FLAGS.weights_ckpt)


if __name__ == '__main__':
    tf.app.run()
