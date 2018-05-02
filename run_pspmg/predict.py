import sys
sys.path.append('../')

import datetime
import os

import tensorflow as tf
from model import pspnet_mg
import numpy as np
import cv2
from database.helper_cityscapes import trainid_to_labelid, coloring
from database.helper_segmentation import *
from database.reader_segmentation import SegmentationImageReader
from experiment_manager.utils import sorted_str_dict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--server', type=int, default=0, help='local machine 0 or server 1 or 2')
parser.add_argument('--epsilon', type=float, default=0.00001, help='epsilon in bn layers')
parser.add_argument('--norm_only', type=int, default=0,
                    help='no beta nor gamma in fused_bn (1). Or with beta and gamma(0).')
parser.add_argument('--data_type', type=int, default=32, help='float32 or float16')
parser.add_argument('--database', type=str, default='CityScapes', help='CityScapes.')
parser.add_argument('--resize_images_method', type=str, default='bilinear', help='resize images method: bilinear or nn')
parser.add_argument('--color_switch', type=int, default=0, help='color switch or not')
parser.add_argument('--test_max_iter', type=int, default=None, help='maximum test iteration')
parser.add_argument('--test_image_size', type=int, default=864,
                    help='spatial size of inputs for test. not used any longer')
parser.add_argument('--mode', type=str, default='val', help='test or val')
parser.add_argument('--structure_in_paper', type=int, default=0, help='structure in paper')

parser.add_argument('--weights_ckpt',
                    type=str,
                    default='./save/1-extra-2/model.ckpt-50000',
                    help='ckpt file for loading the trained weights')
parser.add_argument('--coloring', type=int, default=0, help='coloring the prediction and ground truth.')
parser.add_argument('--mirror', type=int, default=1, help='whether adding the results from mirroring.')
parser.add_argument('--save_prediction', type=int, default=1, help='whether saving prediction.')
parser.add_argument('--ms', type=int, default=0, help='whether applying multi-scale testing.')

FLAGS = parser.parse_args()


def predict(i_ckpt):
    tf.reset_default_graph()

    print '================',
    if FLAGS.data_type == 16:
        print 'using tf.float16 ====================='
        data_type = tf.float16
    else:
        print 'using tf.float32 ====================='
        data_type = tf.float32

    image_size = FLAGS.test_image_size
    print '=====because using pspnet, the inputs have a fixed size and should be divided by 48:', image_size
    assert FLAGS.test_image_size % 48 == 0

    with tf.device('/cpu:0'):
        reader = SegmentationImageReader(
            FLAGS.server,
            FLAGS.database,
            FLAGS.mode,
            (image_size, image_size),
            random_scale=False,
            random_mirror=False,
            random_blur=False,
            random_rotate=False,
            color_switch=FLAGS.color_switch)

    images_pl = [tf.placeholder(tf.float32, [None, image_size, image_size, 3])]
    labels_pl = [tf.placeholder(tf.int32, [None, image_size, image_size, 1])]

    with tf.variable_scope('resnet_v1_101'):
        model = pspnet_mg.PSPNetMG(reader.num_classes, None, None, None,
                                   mode='val', bn_epsilon=FLAGS.epsilon, resnet='resnet_v1_101',
                                   norm_only=FLAGS.norm_only,
                                   float_type=data_type,
                                   has_aux_loss=False,
                                   structure_in_paper=FLAGS.structure_in_paper,
                                   resize_images_method=FLAGS.resize_images_method
                                   )
        l = model.inference(images_pl)
    # ========================= end of building model ================================

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if i_ckpt is not None:
        loader = tf.train.Saver(max_to_keep=0)
        loader.restore(sess, i_ckpt)
        eval_step = i_ckpt.split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' % (i_ckpt, eval_step))

    print '======================= eval process begins ========================='
    if FLAGS.save_prediction == 0 and FLAGS.mode != 'test':
        print 'not saving prediction ... '

    average_loss = 0.0
    confusion_matrix = np.zeros((reader.num_classes, reader.num_classes), dtype=np.int64)

    if FLAGS.save_prediction == 1 or FLAGS.mode == 'test':
        try:
            os.mkdir('./' + FLAGS.mode + '_set')
        except:
            pass
        prefix = './' + FLAGS.mode + '_set'
        try:
            os.mkdir(os.path.join(prefix, FLAGS.weights_ckpt.split('/')[-2]))
        except:
            pass
        prefix = os.path.join(prefix, FLAGS.weights_ckpt.split('/')[-2])

    if FLAGS.ms == 1:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    else:
        scales = [1.0]

    images_filenames = reader.image_list
    labels_filenames = reader.label_list

    if FLAGS.test_max_iter is None:
        max_iter = len(images_filenames)
    else:
        max_iter = FLAGS.test_max_iter

    # IMG_MEAN = [123.680000305, 116.778999329, 103.939002991]  # RGB mean from official PSPNet

    step = 0
    while step < max_iter:
        image, label = cv2.imread(images_filenames[step], 1), cv2.imread(labels_filenames[step], 0)
        label = np.reshape(label, [1, label.shape[0], label.shape[1], 1])
        image_height, image_width = image.shape[0], image.shape[1]

        total_logits = np.zeros((image_height, image_width, reader.num_classes), np.float32)
        for scale in scales:
            imgsplitter = ImageSplitter(image, scale, FLAGS.color_switch, image_size, reader.img_mean)
            crops = imgsplitter.get_split_crops()

            # This is a suboptimal solution. More batches each iter, more rapid.
            # But the limit of batch size is unknown.
            # TODO: Or there should be a more efficient way.
            if crops.shape[0] > 10:
                half = crops.shape[0] / 2

                feed_dict = {images_pl[0]: crops[0:half]}
                [logits_0] = sess.run([
                    model.probabilities
                ],
                    feed_dict=feed_dict
                )

                feed_dict = {images_pl[0]: crops[half:]}
                [logits_1] = sess.run([
                    model.probabilities
                ],
                    feed_dict=feed_dict
                )
                logits = np.concatenate((logits_0, logits_1), axis=0)
            else:
                feed_dict = {images_pl[0]: imgsplitter.get_split_crops()}
                [logits] = sess.run([
                    model.probabilities
                ],
                    feed_dict=feed_dict
                )
            scale_logits = imgsplitter.reassemble_crops(logits)

            if FLAGS.mirror == 1:
                image_mirror = image[:, ::-1]
                imgsplitter_mirror = ImageSplitter(image_mirror, scale, FLAGS.color_switch, image_size, reader.img_mean)
                crops_m = imgsplitter_mirror.get_split_crops()
                if crops_m.shape[0] > 10:
                    half = crops_m.shape[0] / 2

                    feed_dict = {images_pl[0]: crops_m[0:half]}
                    [logits_0] = sess.run([
                        model.probabilities
                    ],
                        feed_dict=feed_dict
                    )

                    feed_dict = {images_pl[0]: crops_m[half:]}
                    [logits_1] = sess.run([
                        model.probabilities
                    ],
                        feed_dict=feed_dict
                    )
                    logits_m = np.concatenate((logits_0, logits_1), axis=0)
                else:
                    feed_dict = {images_pl[0]: imgsplitter_mirror.get_split_crops()}
                    [logits_m] = sess.run([
                        model.probabilities
                    ],
                        feed_dict=feed_dict
                    )
                logits_m = imgsplitter_mirror.reassemble_crops(logits_m)
                scale_logits += logits_m[:, ::-1]

            if scale != 1.0:
                scale_logits = cv2.resize(scale_logits, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

            total_logits += scale_logits

        prediction = np.argmax(total_logits, axis=-1)
        # print np.max(label), np.max(prediction)

        if FLAGS.database == 'CityScapes' and (FLAGS.save_prediction == 1 or FLAGS.mode == 'test'):
            image_prefix = images_filenames[step].split('/')[-1].split('_leftImg8bit.png')[0] + '_' \
                           + FLAGS.weights_ckpt.split('/')[-2]

            cv2.imwrite(os.path.join(prefix, image_prefix + '_prediction.png'), trainid_to_labelid(prediction))
            if FLAGS.coloring == 1:
                color_prediction = coloring(prediction)
                cv2.imwrite(os.path.join(prefix, image_prefix + '_coloring.png'),
                            cv2.cvtColor(color_prediction, cv2.COLOR_BGR2RGB))
        elif FLAGS.database == 'SDB' and (FLAGS.save_prediction == 1 or FLAGS.mode == 'test'):
            image_prefix = images_filenames[step].split('/')[-1] + '_' + FLAGS.weights_ckpt.split('/')[-2]
            cv2.imwrite(os.path.join(prefix, image_prefix + '_prediction.png'), prediction)
        else:
            pass

        step += 1

        compute_confusion_matrix(label, prediction, confusion_matrix)
        if step % 20 == 0:
            print '%s %s] %d / %d. iou updating' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step, max_iter)
            compute_iou(confusion_matrix)
            print average_loss / step

    precision = compute_iou(confusion_matrix)
    coord.request_stop()
    coord.join(threads)

    return average_loss / max_iter, precision


def main(_):
    print(sorted_str_dict(FLAGS.__dict__))

    # ============================================================================
    # ===================== Prediction =========================
    # ============================================================================
    loss, precision = predict(FLAGS.weights_ckpt)
    step = FLAGS.weights_ckpt.split('-')[-1]
    print '%s %s] Step %s Test' % (str(datetime.datetime.now()), str(os.getpid()), step)
    print '\t loss = %.4f, precision = %.4f' % (loss, precision)


if __name__ == '__main__':
    tf.app.run()
