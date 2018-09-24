from __future__ import print_function, division, absolute_import
try:
    xrange
except NameError:
    xrange = range

import datetime
import os

import tensorflow as tf
from model import pspnet_mg
import math
import numpy as np
import cv2
from database.helper_segmentation import *
from database.reader_segmentation import SegmentationImageReader
from experiment_manager.utils import LogDir, sorted_str_dict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=1, help='gpu num')

parser.add_argument('--consider_dilated', type=int, default=0, help='consider dilated conv weights when using L2-SP.')
parser.add_argument('--network', type=str, default='resnet_v1_50', help='resnet_v1_50 or 101')
parser.add_argument('--database', type=str, default='SBD', help='SBD, Cityscapes or ADE.')
parser.add_argument('--subsets_for_training', type=str, default='train,val', help='whether use val set for training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--weight_decay_mode', type=int, default=1, help='weight decay mode')
parser.add_argument('--weight_decay_rate', type=float, default=0.01, help='weight decay rate for existing layers')
parser.add_argument('--weight_decay_rate2', type=float, default=0.01, help='weight decay rate for new layers')
parser.add_argument('--train_image_size', type=int, default=480, help='spatial size of inputs')

# will rarely change
parser.add_argument('--lrn_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--random_rotate', type=int, default=1, help='random rotate')
parser.add_argument('--random_scale', type=int, default=1, help='random scale')
parser.add_argument('--train_max_iter', type=int, default=30000, help='Maximum training iteration')
parser.add_argument('--snapshot', type=int, default=15000, help='snapshot every ')
parser.add_argument('--scale_min', type=float, default=0.5, help='random scale rate min')
parser.add_argument('--scale_max', type=float, default=2.0, help='random scale rate max')

# nearly will never change
parser.add_argument('--color_switch', type=int, default=0, help='color switch or not')
parser.add_argument('--loss_type', type=str, default='normal', help='normal, focal_1, etc.')
parser.add_argument('--structure_in_paper', type=int, default=0, help='first conv layers')
parser.add_argument('--train_like_in_paper', type=int, default=0,
                    help='new layers receive 10 times learning rate; biases * 2')
parser.add_argument('--bn_frozen', type=int, default=0, help='freezing the statistics in existing bn layers')
parser.add_argument('--blur', type=int, default=1, help='random blur: brightness/saturation/constrast')
parser.add_argument('--initializer', type=str, default='he', help='he or xavier')
parser.add_argument('--optimizer', type=str, default='mom', help='mom, sgd, adam, more to be added')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for mom optimizer')
parser.add_argument('--float_type', type=int, default=32, help='float32 or float16')
parser.add_argument('--data_format', type=str, default='NHWC', help='NHWC or NCHW.')
parser.add_argument('--has_aux_loss', type=int, default=1, help='with(1) or without(0) auxiliary loss')
parser.add_argument('--poly_lr', type=int, default=1, help='poly learning rate policy')
parser.add_argument('--new_layer_names', type=str, default=None, help='with(1) or without(0) auxiliary loss')

# does not affect learning process
parser.add_argument('--fine_tune_filename', type=str,
                    default='./z_pretrained_weights/resnet_v1_50.ckpt',
                    help='fine_tune_filename')
parser.add_argument('--resume_step', type=int, default=None, help='resume step')
parser.add_argument('--lr_step', type=str, default=None, help='list of lr rate decreasing step. Default None.')
parser.add_argument('--step_size', type=float, default=0.1,
                    help='Each lr_step, learning rate decreases . Default to 0.1')
parser.add_argument('--save_first_iteration', type=int, default=0, help='whether saving the initial model')

# test or evaluation
parser.add_argument('--eval_only', type=int, default=0, help='only do the evaluation (1) or do train and eval (0).')
parser.add_argument('--test_max_iter', type=int, default=None, help='maximum test iteration')
parser.add_argument('--test_image_size', type=int, default=864,
                    help='spatial size of inputs for test. not used any longer')
parser.add_argument('--mirror', type=int, default=1, help='whether adding the results from mirroring.')
FLAGS = parser.parse_args()


def get_available_gpus(gpu_num):
    # reference: https://stackoverflow.com/a/41638727/4834515
    import subprocess, re

    def run_command(cmd):
        """Run command, return output as string."""
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
        return output.decode("ascii")

    def list_available_gpus():
        """Returns list of available GPU ids."""
        output = run_command("nvidia-smi -L")
        # lines of the form GPU 0: TITAN X
        gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
        result = []
        for line in output.strip().split("\n"):
            m = gpu_regex.match(line)
            assert m, "Couldnt parse " + line
            result.append(int(m.group("gpu_id")))
        return result

    def gpu_memory_map():
        """Returns map of GPU id to memory allocated on that GPU."""

        output = run_command("nvidia-smi")
        gpu_output = output[output.find("GPU Memory"):]
        # lines of the form
        # |    0      8734    C   python                                       11705MiB |
        memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
        rows = gpu_output.split("\n")
        result = range(len(list_available_gpus()))
        for row in rows:
            m = memory_regex.search(row)
            if not m:
                continue
            gpu_id = int(m.group("gpu_id"))
            gpu_memory = int(m.group("gpu_memory"))
            if gpu_memory < 1000:
                continue
            else:
                result.remove(gpu_id)

        return result

    if gpu_num <= 0:
        raise ValueError('GPU number <= 0.')
    results = gpu_memory_map()
    if len(results) < gpu_num:
        raise ValueError('No enough GPUs.')
    if len(results) == 1:
        return str(results[0])

    str_gpus = ''
    for i in xrange(gpu_num-1):
        str_gpus += str(results[i]) + ','
    str_gpus += results[gpu_num-1]

    return str_gpus


def model_id():
    FLAGS_dict = FLAGS.__dict__
    model_id = str(FLAGS_dict['network']) + '-' + str(FLAGS_dict['train_image_size'])
    model_id += '-' + str(FLAGS_dict['subsets_for_training'])
    model_id += '-' + 'L2-SP' if FLAGS_dict['weight_decay_mode'] == 1 else 'L2'
    model_id += '-' + 'wd_alpha' + str(FLAGS_dict['weight_decay_rate'])
    model_id += '-' + 'wd_beta' + str(FLAGS_dict['weight_decay_rate2'])

    model_arguments = ['batch_size', 'lrn_rate', 'consider_dilated', 'random_rotate', 'random_scale']
    for arg in model_arguments:
        model_id += '-' + arg + str(FLAGS_dict[arg])

    return model_id


def train(resume_step=None):
    # < preparing arguments >
    if FLAGS.float_type == 16:
        print('\n< using tf.float16 >\n')
        float_type = tf.float16
    else:
        print('\n< using tf.float32 >\n')
        float_type = tf.float32
    new_layer_names = FLAGS.new_layer_names
    if FLAGS.new_layer_names is not None:
        new_layer_names = new_layer_names.split(',')

    # < data set >
    data_list = FLAGS.subsets_for_training.split(',')
    if len(data_list) < 1:
        data_list = ['train']
    list_images = []
    list_labels = []
    with tf.device('/cpu:0'):
        reader = SegmentationImageReader(
            FLAGS.database,
            data_list,
            (FLAGS.train_image_size, FLAGS.train_image_size),
            FLAGS.random_scale,
            random_mirror=True,
            random_blur=True,
            random_rotate=FLAGS.random_rotate,
            color_switch=FLAGS.color_switch,
            scale_rate=(FLAGS.scale_min, FLAGS.scale_max))
        for _ in xrange(FLAGS.gpu_num):
            image_batch, label_batch = reader.dequeue(FLAGS.batch_size)
            list_images.append(image_batch)
            list_labels.append(label_batch)

    # < network >
    model = pspnet_mg.PSPNetMG(reader.num_classes,
                               mode='train', resnet=FLAGS.network, bn_mode='frozen' if FLAGS.bn_frozen else 'gather',
                               data_format=FLAGS.data_format,
                               initializer=FLAGS.initializer,
                               fine_tune_filename=FLAGS.fine_tune_filename,
                               wd_mode=FLAGS.weight_decay_mode,
                               gpu_num=FLAGS.gpu_num,
                               float_type=float_type,
                               has_aux_loss=FLAGS.has_aux_loss,
                               train_like_in_paper=FLAGS.train_like_in_paper,
                               structure_in_paper=FLAGS.structure_in_paper,
                               new_layer_names=new_layer_names,
                               loss_type=FLAGS.loss_type,
                               consider_dilated=FLAGS.consider_dilated)
    train_ops = model.build_train_ops(list_images, list_labels)

    # < log dir and model id >
    logdir = LogDir(FLAGS.database, model_id())
    logdir.print_all_info()
    if not os.path.exists(logdir.log_dir):
        print('creating ', logdir.log_dir, '...')
        os.mkdir(logdir.log_dir)
    if not os.path.exists(logdir.database_dir):
        print('creating ', logdir.database_dir, '...')
        os.mkdir(logdir.database_dir)
    if not os.path.exists(logdir.exp_dir):
        print('creating ', logdir.exp_dir, '...')
        os.mkdir(logdir.exp_dir)
    if not os.path.exists(logdir.snapshot_dir):
        print('creating ', logdir.snapshot_dir, '...')
        os.mkdir(logdir.snapshot_dir)

    gpu_options = tf.GPUOptions(allow_growth=False)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init)

    # < convert npy to .ckpt >
    step = 0
    if '.npy' in FLAGS.fine_tune_filename:
        # This can transform .npy weights with variables names being the same to the tf ckpt model.
        fine_tune_variables = []
        npy_dict = np.load(FLAGS.fine_tune_filename).item()
        new_layers_names = ['Momentum']
        for v in tf.global_variables():
            if any(elem in v.name for elem in new_layers_names):
                continue

            name = v.name.split(':0')[0]
            if name not in npy_dict:
                continue

            v.load(npy_dict[name], sess)
            fine_tune_variables.append(v)

        saver = tf.train.Saver(var_list=fine_tune_variables)
        saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=0)
        return

    # < load pre-trained model>
    import_variables = tf.trainable_variables()
    if FLAGS.fine_tune_filename is not None and resume_step is None:
        fine_tune_variables = []
        new_layers_names = model.new_layers_names
        new_layers_names.append('Momentum')
        new_layers_names.append('up_sample')
        for v in import_variables:
            if any(elem in v.name for elem in new_layers_names):
                print('< Finetuning Process: not import %s >' % v.name)
                continue
            fine_tune_variables.append(v)

        loader = tf.train.Saver(var_list=fine_tune_variables, allow_empty=True)
        loader.restore(sess, FLAGS.fine_tune_filename)
        print('< Succesfully loaded fine-tune model from %s. >' % FLAGS.fine_tune_filename)
    elif resume_step is not None:
        # ./snapshot/model.ckpt-3000
        i_ckpt = logdir.snapshot_dir + '/model.ckpt-%d' % resume_step

        loader = tf.train.Saver(max_to_keep=0)
        loader.restore(sess, i_ckpt)

        step = resume_step
        print('< Succesfully loaded model from %s at step=%s. >' % (i_ckpt, resume_step))
    else:
        print('< Not import any model. >')

    f_log = open(logdir.exp_dir + '/' + str(datetime.datetime.now()) + '.txt', 'w')
    f_log.write('step,loss,precision,wd\n')
    f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')

    print('\n< training process begins >\n')
    average_loss = 0.0
    show_period = 20
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

    if FLAGS.save_first_iteration == 1:
        saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=step)

    has_nan = False
    while step < max_iter + 1:
        if FLAGS.poly_lr == 1:
            lrn_rate = ((1-1.0*step/max_iter)**0.9) * FLAGS.lrn_rate

        step += 1
        if len(lr_step) > 0 and step == lr_step[0]:
            lrn_rate *= FLAGS.step_size
            lr_step.remove(step)

        _, loss, wd, precision = sess.run([
            train_ops, model.loss, model.wd, model.precision_op
        ],
            feed_dict={
                model.lrn_rate_ph: lrn_rate,
                model.wd_rate_ph: wd_rate,
                model.wd_rate2_ph: wd_rate2
            }
        )

        if math.isnan(loss) or math.isnan(wd):
            print('\nloss or weight norm is nan. Training Stopped!\n')
            has_nan = True
            break

        average_loss += loss

        if step % snapshot == 0:
            saver.save(sess, logdir.snapshot_dir + '/model.ckpt', global_step=step)
            sess.run([tf.local_variables_initializer()])

        if step % show_period == 0:
            left_hours = 0

            if t0 is not None:
                delta_t = (datetime.datetime.now() - t0).seconds
                left_time = (max_iter - step) / show_period * delta_t
                left_hours = left_time/3600.0

            t0 = datetime.datetime.now()
            average_loss /= show_period

            f_log.write('%d,%f,%f,%f\n' % (step, average_loss, precision, wd))
            f_log.flush()

            print('%s %s] Step %s, lr = %f, wd_rate = %f, wd_rate_2 = %f ' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step, lrn_rate, wd_rate, wd_rate2))
            print('\t loss = %.4f, precision = %.4f, wd = %.4f' % (average_loss, precision, wd))
            print('\t estimated time left: %.1f hours. %d/%d' % (left_hours, step, max_iter))

            average_loss = 0.0

    coord.request_stop()
    coord.join(threads)

    return f_log, logdir, has_nan  # f_log and logdir returned for eval.


def eval(i_ckpt):
    # does not perform multi-scale test. ms-test is in predict.py
    tf.reset_default_graph()

    if FLAGS.float_type == 16:
        print('\n< using tf.float16 >\n')
        float_type = tf.float16
    else:
        print('\n< using tf.float32 >\n')
        float_type = tf.float32

    input_size = FLAGS.test_image_size
    with tf.device('/cpu:0'):
        reader = SegmentationImageReader(
            FLAGS.database,
            'val',
            (input_size, input_size),
            random_scale=False,
            random_mirror=False,
            random_blur=False,
            random_rotate=False,
            color_switch=FLAGS.color_switch)

    images_pl = [tf.placeholder(tf.float32, [None, input_size, input_size, 3])]
    labels_pl = [tf.placeholder(tf.int32, [None, input_size, input_size, 1])]

    model = pspnet_mg.PSPNetMG(reader.num_classes,
                               mode='val', resnet=FLAGS.network,
                               data_format=FLAGS.data_format,
                               float_type=float_type,
                               has_aux_loss=False,
                               structure_in_paper=FLAGS.structure_in_paper)
    logits = model.inference(images_pl)
    probas_op = tf.nn.softmax(logits, dim=1 if FLAGS.data_format == 'NCHW' else 3)
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

    print('< eval process begins >')
    average_loss = 0.0
    confusion_matrix = np.zeros((reader.num_classes, reader.num_classes), dtype=np.int64)

    images_filenames = reader.image_list
    labels_filenames = reader.label_list
    img_mean = reader.img_mean

    if FLAGS.test_max_iter is None:
        max_iter = len(images_filenames)
    else:
        max_iter = FLAGS.test_max_iter

    step = 0
    while step < max_iter:
        image, label = cv2.imread(images_filenames[step], 1), cv2.imread(labels_filenames[step], 0)
        label = np.reshape(label, [1, label.shape[0], label.shape[1], 1])

        imgsplitter = ImageSplitter(image, 1.0, FLAGS.color_switch, input_size, img_mean)
        feed_dict = {images_pl[0]: imgsplitter.get_split_crops()}
        [logits] = sess.run([
            probas_op
        ],
            feed_dict=feed_dict
        )
        total_logits = imgsplitter.reassemble_crops(logits)
        if FLAGS.mirror == 1:
            image_mirror = image[:, ::-1]
            imgsplitter_mirror = ImageSplitter(image_mirror, 1.0, FLAGS.color_switch, input_size, img_mean)
            feed_dict = {images_pl[0]: imgsplitter_mirror.get_split_crops()}
            [logits_m] = sess.run([
                probas_op
            ],
                feed_dict=feed_dict
            )
            logits_m = imgsplitter_mirror.reassemble_crops(logits_m)
            total_logits += logits_m[:, ::-1]

        prediction = np.argmax(total_logits, axis=-1)
        step += 1
        compute_confusion_matrix(label, prediction, confusion_matrix)
        if step % 20 == 0:
            print('%s %s] %d / %d. iou updating' \
                  % (str(datetime.datetime.now()), str(os.getpid()), step, max_iter))
            compute_iou(confusion_matrix)
            print('imprecise loss', average_loss / step)

    precision = compute_iou(confusion_matrix)
    coord.request_stop()
    coord.join(threads)

    return average_loss / max_iter, precision


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu = str(get_available_gpus(FLAGS.gpu_num))
    print(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # ============================================================================
    # ============================= TRAIN ========================================
    # ============================================================================
    print(sorted_str_dict(FLAGS.__dict__))
    if FLAGS.resume_step is not None:
        print('Ready to resume from step %d.' % FLAGS.resume_step)

    assert FLAGS.gpu_num is not None, 'should specify the number of gpu.'
    assert FLAGS.gpu_num > 0, 'the number of gpu should be bigger than 0.'
    if FLAGS.eval_only:
        logdir = LogDir(FLAGS.database, model_id())
        logdir.print_all_info()
        f_log = open(logdir.exp_dir + '/' + str(datetime.datetime.now()) + '.txt', 'w')
        f_log.write('step,loss,precision,wd\n')
        f_log.write(sorted_str_dict(FLAGS.__dict__) + '\n')
    else:
        f_log, logdir, has_nan = train(FLAGS.resume_step)

        if has_nan:
            f_log.write('TEST:0,nan,nan\n')
            f_log.flush()
            return

    # ============================================================================
    # ============================= EVAL =========================================
    # ============================================================================
    f_log.write('TEST:step,loss,precision\n')

    import glob
    i_ckpts = sorted(glob.glob(logdir.snapshot_dir + '/model.ckpt-*.index'), key=os.path.getmtime)

    # ============================================================================
    # ======================== Eval for the last model ===========================
    # ============================================================================
    i_ckpt = i_ckpts[-1].split('.index')[0]
    loss, precision = eval(i_ckpt)
    step = i_ckpt.split('-')[-1]
    print('%s %s] Step %s Test' % (str(datetime.datetime.now()), str(os.getpid()), step))
    print('\t loss = %.4f, precision = %.4f' % (loss, precision))
    f_log.write('TEST:%s,%f,%f\n' % (step, loss, precision))
    f_log.flush()

    f_log.close()


if __name__ == '__main__':
    tf.app.run()
