import tensorflow as tf
import os
import datetime
import math
import numpy as np

from args import FLAGS
from database import reader, helper
from model import pspnet_mg
from experiment_manager.utils import prepare_log_dir, sorted_str_dict


def gpu_num():
    return len(FLAGS.visible_gpus.split(','))


def chunks(l, n):
    if len(l) < n:
        return list(range(len(l)+1))

    splitters = list(range(0, len(l) - len(l) % n, max(len(l) // n, 1)))
    for i in range(len(splitters)):
        splitters[i] += min(len(l) % n, i)
    splitters.append(len(l))

    return splitters


def get_model_id():
    FLAGS_dict = FLAGS.__dict__
    model_id = FLAGS_dict['prefix'] + '-' + str(FLAGS_dict['network'])
    model_id += '-gpu_num' + str(gpu_num())

    model_arguments = ['batch_size', 'lrn_rate', 'random_scale', 'random_rotate']
    for arg in model_arguments:
        model_id += '-' + arg + str(FLAGS_dict[arg])

    model_arguments_no_name = ['train_image_size', 'train_max_iter', 'subsets_for_training',
                               'weight_decay_mode', 'weight_decay_rate', 'weight_decay_rate2',
                               'train_like_in_caffe', 'three_convs_beginning', 'random_mirror', 'random_blur']
    for arg in model_arguments_no_name:
        model_id += '-' + str(FLAGS_dict[arg])

    if FLAGS_dict['new_layer_names'] is not None:
        model_id += '-' + str(FLAGS_dict['new_layer_names'])

    return model_id


def train_and_eval():
    # < data set >
    data_list = FLAGS.subsets_for_training.split(',')
    if len(data_list) < 1:
        data_list = ['train']

    train_reader_inits = []
    eval_reader_inits = []
    with tf.device('/cpu:0'):
        if FLAGS.reader_method == 'queue':
            train_image_reader = reader.QueueBasedImageReader(FLAGS.database, data_list)
            batch_images, batch_labels = train_image_reader.get_batch(FLAGS.batch_size * gpu_num(),
                                                                      FLAGS.train_image_size,
                                                                      FLAGS.random_mirror,
                                                                      FLAGS.random_blur,
                                                                      FLAGS.random_rotate,
                                                                      FLAGS.color_switch,
                                                                      FLAGS.random_scale,
                                                                      (FLAGS.scale_min, FLAGS.scale_max))
            list_images = tf.split(batch_images, gpu_num())
            list_labels = tf.split(batch_labels, gpu_num())

            eval_image_reader = reader.QueueBasedImageReader(FLAGS.database, 'val')
            eval_image, eval_label, _ = eval_image_reader.get_eval_batch(FLAGS.color_switch)
        else:
            # the performance is not good as using queue runners.
            train_image_reader = reader.ImageReader(FLAGS.database, data_list)
            train_reader_iterator = train_image_reader.get_batch_iterator(FLAGS.batch_size * gpu_num(),
                                                                          FLAGS.train_image_size,
                                                                          FLAGS.random_mirror,
                                                                          FLAGS.random_blur,
                                                                          FLAGS.random_rotate,
                                                                          FLAGS.color_switch,
                                                                          FLAGS.random_scale,
                                                                          (FLAGS.scale_min, FLAGS.scale_max))
            batch_images, batch_labels = train_reader_iterator.get_next()
            list_images = tf.split(batch_images, gpu_num())
            list_labels = tf.split(batch_labels, gpu_num())

            eval_image_reader = reader.ImageReader(FLAGS.database, 'val')
            eval_reader_iterator = eval_image_reader.get_eval_iterator(FLAGS.color_switch)
            eval_image, eval_label, _ = eval_reader_iterator.get_next()  # one image.

            train_reader_inits.append(train_reader_iterator.initializer)
            eval_reader_inits.append(eval_reader_iterator.initializer)

    # < network >
    model = pspnet_mg.PSPNetMG(train_image_reader.num_classes, FLAGS.network, gpu_num(), FLAGS.initializer,
                               FLAGS.weight_decay_mode, FLAGS.fine_tune_filename, FLAGS.optimizer, FLAGS.momentum,
                               FLAGS.train_like_in_caffe, FLAGS.three_convs_beginning, FLAGS.new_layer_names,
                               consider_dilated=FLAGS.consider_dilated)
    train_ops, losses_op, metrics_op = model.build_train_ops(list_images, list_labels)

    eval_image_pl = []
    crop_size = FLAGS.test_image_size
    for _ in range(gpu_num()):
        eval_image_pl.append(tf.placeholder(tf.float32, [None, crop_size, crop_size, 3]))
    eval_probas_op = model.build_forward_ops(eval_image_pl)

    # < log dir and model id >
    exp_dir, snapshot_dir = prepare_log_dir(FLAGS.database, get_model_id())

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)

    if FLAGS.reader_method == 'queue':
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    init = [tf.global_variables_initializer(), tf.local_variables_initializer()] + train_reader_inits
    sess.run(init)

    # < load pre-trained model>
    import_variables = tf.trainable_variables()
    if FLAGS.fine_tune_filename is not None:
        fine_tune_variables = []
        new_layers_names = model.new_layers_names
        new_layers_names.append('Momentum')
        new_layers_names.append('up_sample')
        for v in import_variables:
            if any(elem in v.name for elem in new_layers_names):
                print('\t[verbo] < Finetuning Process: not import %s >' % v.name)
                continue
            fine_tune_variables.append(v)

        loader = tf.train.Saver(var_list=fine_tune_variables, allow_empty=True)
        loader.restore(sess, FLAGS.fine_tune_filename)
        print('\t[verbo] < Succesfully loaded fine-tune model from %s. >' % FLAGS.fine_tune_filename)
    else:
        print('\t[verbo] < Not import any model. >')

    f_log = open(exp_dir + '/' + str(datetime.datetime.now()) + '.txt', 'w')
    tags = ''
    for loss_op in losses_op:
        tags += loss_op.name.split('/')[-1].split(':')[0] + ','
    for metric_op in metrics_op:
        tags += metric_op.name.split('/')[-1].split(':')[0] + ','
    tags = tags[:-1]
    f_log.write(tags + '\n')
    f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')

    print('\n\t < training process begins >\n')
    show_period = FLAGS.train_max_iter // 2000
    snapshot = FLAGS.snapshot
    max_iter = FLAGS.train_max_iter
    lrn_rate = FLAGS.lrn_rate

    lr_step = []
    if FLAGS.lr_step is not None:
        temps = FLAGS.lr_step.split(',')
        for t in temps:
            lr_step.append(int(t))

    saver = tf.train.Saver(max_to_keep=2)
    t0 = None
    wd_rate = FLAGS.weight_decay_rate
    wd_rate2 = FLAGS.weight_decay_rate2
    has_nan = False
    step = 0

    if FLAGS.save_first_iteration == 1:
        saver.save(sess, snapshot_dir + '/model.ckpt', global_step=step)

    def run_for_eval(input_image):
        H, W, channel = input_image.shape

        # < in case that input_image is smaller than crop_size >
        dif_height = H - crop_size
        dif_width = W - crop_size
        if dif_height < 0 or dif_width < 0:
            input_image = helper.numpy_pad_image(input_image, dif_height, dif_width)
            H, W, channel = input_image.shape

        # < split >
        split_crops = []
        heights = helper.decide_intersection(H, crop_size)
        widths = helper.decide_intersection(W, crop_size)
        for height in heights:
            for width in widths:
                image_crop = input_image[height:height + crop_size, width:width + crop_size]
                split_crops.append(image_crop[np.newaxis, :])

        feed_dict = {}
        splitters = chunks(split_crops, gpu_num())
        for list_index in range(len(splitters) - 1):
            piece_crops = np.concatenate(split_crops[splitters[list_index]: splitters[list_index + 1]])
            feed_dict[eval_image_pl[list_index]] = piece_crops

        for i in range(gpu_num()):
            if eval_image_pl[i] not in feed_dict.keys():
                feed_dict[eval_image_pl[i]] = np.zeros((1, crop_size, crop_size, 3), np.float32)

        proba_crops_pieces = sess.run(eval_probas_op, feed_dict=feed_dict)
        proba_crops = np.concatenate(proba_crops_pieces)

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

    while step < max_iter + 1:
        if FLAGS.poly_lr == 1:
            lrn_rate = ((1-1.0*step/max_iter)**0.9) * FLAGS.lrn_rate

        step += 1
        if len(lr_step) > 0 and step == lr_step[0]:
            lrn_rate *= FLAGS.step_size
            lr_step.remove(step)

        _, losses, metrics = sess.run([train_ops, losses_op, metrics_op],
                                      feed_dict={
                                          model.lrn_rate_ph: lrn_rate,
                                          model.wd_rate_ph: wd_rate,
                                          model.wd_rate2_ph: wd_rate2})

        if math.isnan(losses[0]) or math.isnan(losses[-1]):
            print('\nloss or weight norm is nan. Training Stopped!\n')
            has_nan = True
            break

        if step % show_period == 0:
            left_hours = 0
            if t0 is not None:
                delta_t = (datetime.datetime.now() - t0).total_seconds()
                left_time = (max_iter - step) / show_period * delta_t
                left_hours = left_time / 3600.0
            t0 = datetime.datetime.now()

            # these losses are not averaged.
            merged_losses = losses + metrics
            str_merged_loss = str(step) + ','
            for i, l in enumerate(merged_losses):
                if i == len(merged_losses) - 1:
                    str_merged_loss += str(l) + '\n'
                else:
                    str_merged_loss += str(l) + ','
            f_log.write(str_merged_loss)
            f_log.flush()

            print('%s %s] Step %d, lr = %f, wd_mode = %d, wd_rate = %f, wd_rate_2 = %f '
                  % (str(datetime.datetime.now()), str(os.getpid()), step, lrn_rate,
                     FLAGS.weight_decay_mode, wd_rate, wd_rate2))
            for i, tag in enumerate(tags.split(',')):
                print(tag, '=', merged_losses[i], end=', ')
            print('')
            print('\tEstimated time left: %.2f hours. %d/%d' % (left_hours, step, max_iter))

        if step % snapshot == 0 or step == max_iter:
            saver.save(sess, snapshot_dir + '/model.ckpt', global_step=step)
            confusion_matrix = np.zeros((eval_image_reader.num_classes, eval_image_reader.num_classes),
                                        dtype=np.int64)
            sess.run([tf.local_variables_initializer()] + eval_reader_inits)
            for i in range(len(eval_image_reader.image_list)):
                orig_one_image, one_label = sess.run([eval_image, eval_label])
                proba = run_for_eval(orig_one_image)
                prediction = np.argmax(proba, axis=-1)
                helper.compute_confusion_matrix(one_label, prediction, confusion_matrix)

            mIoU = helper.compute_iou(confusion_matrix)
            str_merged_loss = 'TEST:' + str(step) + ',' + str(mIoU) + '\n'
            f_log.write(str_merged_loss)
            f_log.flush()

    f_log.close()

    if FLAGS.reader_method == 'queue':
        coord.request_stop()
        coord.join(threads)


def main(_):
    print(sorted_str_dict(FLAGS.__dict__))
    assert FLAGS.visible_gpus is not None, 'should specify the number of gpu.'
    assert gpu_num() > 0, 'the number of gpu should be bigger than 0.'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_gpus

    train_and_eval()


if __name__ == '__main__':
    tf.app.run()
