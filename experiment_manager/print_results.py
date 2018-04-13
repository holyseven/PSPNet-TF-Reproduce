from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
from experiment_manager import utils
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='run_classification', help='run_classification or run_segmentation')
FLAGS = parser.parse_args()

if __name__ == '__main__':
    logreader = utils.LogReader('../' + FLAGS.exp + '/log')
    filter_dict = {'database': 'dogs120'}
    # filter_dict = {'weight_decay_mode': 1}
    logreader.print_necessary_logs(utils.list_toprint, filter_dict)
