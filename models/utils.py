#!/usr/bin/python3.6
#encoding=utf-8
import os, time
import numpy as np
import torch
import torch.nn.functional as F

class Logger(object):
    def __init__(self, fp):
        self.fp = fp

    def __call__(self, string, end='\n'):
        new_string = '[%s] ' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + string
        print(new_string, end=end)
        if self.fp is not None:
            self.fp.write('%s%s' % (new_string, end))

def split_batch(batches, batch_size=160):
    batches_list = []
    tmp_list = []
    for i, batch in enumerate(batches):
        tmp_list.append(batch)
        if i % batch_size == batch_size - 1:
            batches_list.append(tmp_list)
            tmp_list = []
    if tmp_list:
        last_batch = np.random.choice(tmp_list, batch_size)
        batches_list.append(last_batch)
    return batches_list

def to_categorical(label, class_num):
    return np.eye(class_num, dtype=np.float32)[label]

def eval_pr(predict_y, predict_y_prob, y_given):
    positive_num = np.sum(y_given > 0)
    idx = np.argsort(predict_y_prob)[::-1]
    all_pre = [0.]
    all_rec = [0.]
    tp = 0
    fp = 0
    fn = 0
    for i in range(y_given.shape[0]):
        label = y_given[idx[i]]
        pred = predict_y[idx[i]]
        if label == 0:
            if pred > 0:
                fp += 1
        else:
            if pred == 0:
                fn += 1
            else:
                if pred == label:
                    tp += 1
        if (tp + fp) == 0:
            precision = 1.0
        else:
            precision = tp / (tp + fp)
        recall = tp / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
    return all_pre[1:], all_rec[1:], (tp, fp, fn, positive_num)

def eval_pr_ATT(prob, one_hot, y, n=2000):
    positive_num = np.sum(y>0)
    idx = np.argsort(prob)[::-1]
    if idx.shape[0] > n:
        idx = idx[:n]
    correct = 0
    all_pre = [0.]
    all_rec = [0.]
    for count, i in enumerate(idx):
        if one_hot[i] != 0:
            correct += 1
        precision = correct / (count + 1)
        recall = correct / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
    return all_pre[1:], all_rec[1:]

def save_pr(output_dir, epoch, pre, rec):
    fp = open(os.path.join(output_dir, "%d_pr.txt" % (epoch+1)), 'w')
    for p, r in zip(pre, rec):
        fp.write('%f %f\n' % (p, r))
    fp.close()

def idx_relation_dict(filename):
    idx_relation_dict = {}
    fp = open(filename, 'r')
    while True:
        line = fp.readline().strip()
        if not line:
            break
        relation_str = line.split()[0]
        relation_idx = int(line.split()[1])
        idx_relation_dict[relation_idx] = relation_str
    fp.close()
    return idx_relation_dict

def parse_args_name(argv, name):
    i = 1
    while i < len(argv):
        if argv[i] == '--%s' % name:
            return argv[i + 1]
        i += 2
    return None