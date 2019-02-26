import os
import glob
import ast

list_toprint = ['fine_tune_filename', 'database', 'lrn_rate', 'train_max_iter', 'weight_decay_mode',
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


def prepare_log_dir(database, log_dir):
    base_log_dir = './log'
    database_dir = os.path.join(base_log_dir, database)
    exp_dir = os.path.join(database_dir, log_dir)
    snapshot_dir = os.path.join(exp_dir, 'snapshot')

    # < make directories >
    if not os.path.exists(base_log_dir):
        # print('creating ', base_log_dir, '...')
        os.mkdir(base_log_dir)
    if not os.path.exists(database_dir):
        # print('creating ', database_dir, '...')
        os.mkdir(database_dir)
    if not os.path.exists(exp_dir):
        # print('creating ', exp_dir, '...')
        os.mkdir(exp_dir)
    if not os.path.exists(snapshot_dir):
        # print('creating ', snapshot_dir, '...')
        os.mkdir(snapshot_dir)

    return exp_dir, snapshot_dir


def read_and_arrange_logs():
    # could be better
    databases_dir = glob.glob('../log/*')
    for database_dir in databases_dir:
        logs_dir = sorted(glob.glob(database_dir + '/*'))
        all_results = open(database_dir + '/all.results', 'w')
        for log_dir in logs_dir:
            results = []
            logs = glob.glob(log_dir + '/*.txt')
            if len(logs) == 0:
                continue
            for log_filename in logs:
                f_log = open(log_filename)
                lines = f_log.readlines()
                if len(lines) < 2:  # no hyperp dict
                    continue
                if 'TEST' not in lines[-1]:  # no TEST data
                    continue

                result = lines[-1].split('\n')[0].split(',')[-1]
                try:
                    float(result)
                except:
                    continue
                if float(result) > 0.0:
                    results.append(result)
            all_results.writelines(log_dir + ':' + str(results) + '\n')


if __name__ == '__main__':
    read_and_arrange_logs()