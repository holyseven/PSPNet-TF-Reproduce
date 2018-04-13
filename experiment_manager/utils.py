from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import ast

list_toprint = ['network', 'fine_tune_filename', 'database', 'examples_per_class', 'lrn_rate', 'train_max_iter', 'weight_decay_mode',
                'weight_decay_rate', 'weight_decay_rate2', 'batch_size', 'train_image_size', 'initializer']


def sorted_str_dict(dict):
    res = '{'
    keys = sorted(dict.keys())
    for i, k in enumerate(keys):
        if type(dict[k]) == str:
            res += '\'' + str(k) + '\': ' + '\'' + str(dict[k]) + '\''
        else:
            res += '\'' + str(k) + '\': ' + str(dict[k])
        if i < len(keys) - 1:
            res += ', '
    res += '}'
    return res


class LogDir(object):
    def __init__(self, database, log_dir, weight_decay_mode):
        self.log_dir = './log'
        self.database_dir = os.path.join(self.log_dir, database)
        self.exp_dir = os.path.join(self.database_dir, log_dir+'-'+str(weight_decay_mode))
        self.snapshot_dir = os.path.join(self.exp_dir, 'snapshot')

    def print_all_info(self):
        print('============================================')
        print('=============== LogDir Info ================')
        print('log_dir', self.log_dir)
        print('database_dir', self.database_dir)
        print('exp_dir', self.exp_dir)
        print('snapshot_dir', self.snapshot_dir)
        print('=============== LogDir Info ================')
        print('============================================')


class ExpLog(object):
    def __init__(self, hyperp_dict, best_precision, best_loss, best_position, logfile_path):
        self.hyperp_dict = hyperp_dict
        self.best_precision = best_precision
        self.best_loss = best_loss
        self.best_position = best_position
        self.logfile_path = logfile_path

        if self.logfile_path is not list:
            self.logfile_path = [logfile_path]
        if best_precision is not list:
            self.best_precision = [best_precision]
        if best_loss is not list:
            self.best_loss = [best_loss]
        if best_position is not list:
            self.best_position = [best_position]

    def issame(self, explog):
        return self.hyperp_dict == explog.hyperp_dict

    def get_same_hyperp(self, explog, pre_hyperp=None):
        same_hyperp = dict()
        if pre_hyperp is None:
            pre_hyperp = self.hyperp_dict

        for k in pre_hyperp:
            if k in explog.hyperp_dict:
                if explog.hyperp_dict[k] == pre_hyperp[k]:
                    same_hyperp[k] = pre_hyperp[k]
        return same_hyperp

    def add_a_same_explog(self, explog):
        if not self.issame(explog):
            return self

        self.logfile_path += explog.logfile_path
        self.best_precision += explog.best_precision
        self.best_loss += explog.best_loss
        self.best_position += explog.best_position

        return self

    def printall(self):
        if self.best_loss < 0 and self.best_precision == 0:
            return
        print('=============== Exp Log Info ===============')
        print(self.logfile_path)
        print(self.hyperp_dict)
        print('precision: ', self.best_precision)
        print('loss: ', self.best_loss)
        print('position: ', self.best_position)

    def print_some(self, list_toprint, filters):
        if filters is not None:
            for e in filters:
                if filters[e] != self.hyperp_dict[e]:
                    return

        for e in list_toprint:
            print(self.hyperp_dict.get(e, '?'), '\t', end='')
        print(self.best_precision, '\t', self.best_loss)


class LogReader(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.keys_not_important = ['test_batch_size', 'log_dir', 'save_first_iteration', 'snapshot', 'resume_step']
        self.log_paths = []
        self.exp_logs = []
        self.collect_logs(self.log_dir)

    def resume_results(self, log_filename):
        f_log = open(log_filename)
        lines = f_log.readlines()
        if len(lines) < 2:  # no hyperp dict
            return
        hyperp_dict = ast.literal_eval(lines[1])
        for key in self.keys_not_important:
            if key in hyperp_dict:
                hyperp_dict.pop(key)

        if 'TEST' not in lines[-1]:  # no TEST data
            return

        line_index = -1
        best_precision = 0
        best_loss = -1
        best_position = 0
        while True:
            if 'TEST' not in lines[line_index]:
                break

            line = lines[line_index].split('\n')[0]  # line without \n
            items = line.split(',')
            if not items[-1].isalpha():
                if float(items[-1]) > best_precision:
                    best_precision = float(items[-1])
                    best_loss = float(items[-2])
                    best_position = int(items[-3].split(':')[-1])
            else:
                if items[-1] == 'nan':
                    best_precision = float(items[-1])
                    best_loss = float(items[-2])
                    best_position = int(items[-3].split(':')[-1])
                    return ExpLog(hyperp_dict, best_precision, best_loss, best_position, log_filename)

            line_index -= 1

        if best_loss < 0 and best_precision == 0:
            return

        return ExpLog(hyperp_dict, best_precision, best_loss, best_position, log_filename)

    def collect_logs(self, log_dir):
        for database_dir in glob.glob(log_dir+'/*'):
            for exp_dir in glob.glob(database_dir+'/*'):
                for log_path in glob.glob(exp_dir+'/*.txt'):
                    explog = self.resume_results(log_path)
                    if explog is None:
                        continue

                    has_same_exp = False
                    iter_index = 0
                    while iter_index < len(self.exp_logs):
                        if self.exp_logs[iter_index].issame(explog):
                            self.exp_logs[iter_index].add_a_same_explog(explog)
                            has_same_exp = True
                            break
                        iter_index += 1
                    if not has_same_exp:
                        self.exp_logs.append(explog)

                    self.log_paths.append(log_path)

        return self.log_paths

    def print_all_logs(self):
        for i in range(len(self.exp_logs)):
            self.exp_logs[i].printall()

    def print_necessary_logs(self, list_to_print, filters=None):
        for exp_log in sorted(self.exp_logs, key=lambda x: os.stat(x.logfile_path[0]).st_mtime):
            exp_log.print_some(list_to_print, filters)

    def print_different_hyperp(self, passing_filters=None, stop_filters=None):
        # filters
        # passing_filters: must this key and same value for printing out.
        # stop_filters: if have this key and same value, not print out.

        logs_to_print = []
        if passing_filters is not None:
            print('filters: ', passing_filters)
            for i in range(len(self.exp_logs)):
                filtered = False
                for k in passing_filters:
                    if k not in self.exp_logs[i].hyperp_dict:
                        filtered = True
                        break
                    if self.exp_logs[i].hyperp_dict[k] != passing_filters[k]:
                        filtered = True
                        break
                if stop_filters is not None:
                    for k in stop_filters:
                        if k in self.exp_logs[i].hyperp_dict and self.exp_logs[i].hyperp_dict[k] == stop_filters[k]:
                            filtered = True
                            break
                if filtered:
                    continue

                logs_to_print.append(self.exp_logs[i])
        else:
            for i in range(len(self.exp_logs)):
                logs_to_print.append(self.exp_logs[i])

        # find hyperparameters that are the same
        if len(logs_to_print) > 1:
            pre_hyperp = logs_to_print[0].get_same_hyperp(logs_to_print[1])
        else:
            pre_hyperp = logs_to_print[0].hyperp_dict

        for i in range(len(logs_to_print)):
            pre_hyperp = logs_to_print[0].get_same_hyperp(logs_to_print[i], pre_hyperp)

        print('same hyperparameters: ', pre_hyperp)
        print('=============== Exp Log Info ===============')
        for i in range(len(logs_to_print)):
            print(logs_to_print[i].logfile_path)
            print('{',)
            for k in sorted(logs_to_print[i].hyperp_dict):
                if k in pre_hyperp:
                    continue

                print('\'%s\':'%k, logs_to_print[i].hyperp_dict[k], ',', end='')
            print('}')
            print('precision: ', logs_to_print[i].best_precision)
            print('loss: ', logs_to_print[i].best_loss)
            print('position: ', logs_to_print[i].best_position)
