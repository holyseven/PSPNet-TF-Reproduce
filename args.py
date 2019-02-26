import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--visible_gpus', type=str, default='0', help='0,1,2')

# < reader_method: 'queue' is faster for training based on my experiments. >
parser.add_argument('--reader_method', type=str, default='queue', help='reading *images* directly or *queue*')
parser.add_argument('--database', type=str, default='Cityscapes', help='SBD, Cityscapes or ADE.')
parser.add_argument('--network', type=str, default='resnet_v1_101', help='resnet_v1_50 or 101')
parser.add_argument('--subsets_for_training', type=str, default='train', help='whether use val set for training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--weight_decay_mode', type=int, default=1, help='weight decay mode')
parser.add_argument('--weight_decay_rate', type=float, default=0.0001, help='weight decay rate for existing layers')
parser.add_argument('--weight_decay_rate2', type=float, default=0.0001, help='weight decay rate for new layers')
parser.add_argument('--train_image_size', type=int, default=480, help='spatial size of inputs')
parser.add_argument('--fine_tune_filename', type=str,
                    default='./z_pretrained_weights/resnet_v1_101.ckpt',
                    help='fine_tune_filename')
parser.add_argument('--new_layer_names', type=str, default=None, help='new layers names.')
parser.add_argument('--train_max_iter', type=int, default=30000, help='Maximum training iteration')
parser.add_argument('--snapshot', type=int, default=15000, help='snapshot every ')

# will rarely change
parser.add_argument('--consider_dilated', type=int, default=0, help='consider dilated conv weights when using L2-SP.')
parser.add_argument('--lrn_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--random_blur', type=int, default=1, help='random blur: brightness/saturation/constrast')
parser.add_argument('--random_mirror', type=int, default=1, help='random mirror')
parser.add_argument('--random_rotate', type=int, default=1, help='random rotate')
parser.add_argument('--random_scale', type=int, default=1, help='random scale')
parser.add_argument('--scale_min', type=float, default=0.5, help='random scale rate min')
parser.add_argument('--scale_max', type=float, default=2.0, help='random scale rate max')

# nearly will never change
parser.add_argument('--color_switch', type=int, default=0, help='color switch or not')
parser.add_argument('--three_convs_beginning', type=int, default=0, help='first three conv layers')
parser.add_argument('--train_like_in_caffe', type=int, default=0,
                    help='new layers receive 10 times learning rate; biases * 2')
parser.add_argument('--initializer', type=str, default='he', help='he or xavier')
parser.add_argument('--optimizer', type=str, default='mom', help='mom, sgd, adam, more to be added')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for mom optimizer')
parser.add_argument('--poly_lr', type=int, default=1, help='poly learning rate policy')
parser.add_argument('--resume_step', type=int, default=None, help='resume step')
parser.add_argument('--lr_step', type=str, default=None, help='list of lr rate decreasing step. Default None.')
parser.add_argument('--step_size', type=float, default=0.1,
                    help='Each lr_step, learning rate decreases . Default to 0.1')
parser.add_argument('--save_first_iteration', type=int, default=0, help='whether saving the initial model')

# test or evaluation
# network, database, color_switch, visible_gpus
parser.add_argument('--weights_ckpt',
                    type=str,
                    default='/home/jacques/tf_workspace/PSPNet-TF/log/Cityscapes/old/model.ckpt-50000',
                    help='ckpt file for loading the trained weights')
parser.add_argument('--test_subset', type=str, default='val', help='test or val')
parser.add_argument('--test_image_size', type=int, default=864,
                    help='spatial size of inputs for test. not used any longer')
parser.add_argument('--ms', type=int, default=0, help='whether applying multi-scale testing.')
parser.add_argument('--mirror', type=int, default=1, help='whether adding the results from mirroring.')
parser.add_argument('--coloring', type=int, default=1, help='coloring the prediction and ground truth.')

FLAGS = parser.parse_args()
