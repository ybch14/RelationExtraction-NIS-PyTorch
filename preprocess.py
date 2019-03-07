#!/usr/bin/python3.6
#encoding=utf-8
import os
import struct
import pickle
import argparse
import numpy as np
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))
from utils import Logger

idx_word_dict = {0: '0', 1: 'UNK'}
word_idx_dict = {'0': 0, 'UNK': 1}
word_embedding_matrix = None
idx_relation_dict = {}
relation_idx_dict = {}
bags_train = {}
bags_test = {}

class InstanceBag(object):
    def __init__(self, entity_pair, relation, instance_count, sentences, positions, entity_pos):
        self.entity_pair = entity_pair
        self.relation = relation
        self.instance_count = instance_count
        self.sentences = sentences
        self.positions = positions
        self.entity_pos = entity_pos

    def decompose(self):
        return [self.entity_pair, self.relation, self.instance_count, self.sentences, self.positions, self.entity_pos]

    def __len__(self):
        return self.instance_count

    def __str__(self):
        string = "InstanceBag [entity_pair=[%s, %s], "
        "relation=%d, instance_count=%d, sentences=%s, "
        "positions=%s, entity_pos=%s]" \
            % (self.entity_pair[0], self.entity_pair[1], \
               self.relation, self.instance_count, str(self.sentences), \
               str(self.positions), str(self.entity_pos))
    def __repr__(self):
        return self.__str__()

def load_wordvector(input_filename):
    global word_embedding_matrix
    fp = open(input_filename, 'rb')
    line = fp.readline().strip()
    word_count, word_dim = list(map(int, line.split()))
    # 0 for padding index, 1 for UNK word, word start from 2
    word_embedding_matrix = np.zeros((word_count + 2 + int(word_count/2), word_dim), dtype=np.float32)
    word_embedding_matrix[1, :] = np.random.uniform(low=-0.1, high=0.1, size=word_dim)
    count = 1
    while True:
        count += 1
        word = b''
        while True:
            c = fp.read(1)
            if (c == b'') or (c == b' '):
                break
            word += c
        word = word.strip().decode('utf-8')
        buffer = fp.read(4 * word_dim)
        if not buffer:
            break
        vector = np.array(struct.unpack('f' * word_dim, buffer))
        vector /= np.sqrt(np.sum(vector * vector))
        idx_word_dict[count] = word
        word_idx_dict[word] = count
        word_embedding_matrix[count, :] = vector
    fp.close()

def load_relation(filename):
    fp = open(filename, 'r')
    while True:
        line = fp.readline().strip()
        if not line:
            break
        relation_str = line.split()[0]
        relation_idx = int(line.split()[1])
        idx_relation_dict[relation_idx] = relation_str
        relation_idx_dict[relation_str] = relation_idx
    fp.close()

def load_train_and_test_data(train_filename, test_filename):
    global word_embedding_matrix
    word_count = max(list(word_idx_dict.values()))
    word_dim = word_embedding_matrix.shape[1]
    fp = open(train_filename, 'r')
    while True:
        line = fp.readline().strip()
        if not line:
            break
        e1, e2, head_s, tail_s, relation_str, sentence = line.split('\t')
        entity_pair = [head_s, tail_s]
        if word_idx_dict.get(head_s, 1) == 1: # unknown head entity
            word_count += 1
            idx_word_dict[word_count] = head_s
            word_idx_dict[head_s] = word_count
            word_embedding_matrix[word_count, :] = np.random.uniform(low=-0.1, high=0.1, size=word_dim)
        if word_idx_dict.get(tail_s, 1) == 1: # unknown tail entity
            word_count += 1
            idx_word_dict[word_count] = tail_s
            word_idx_dict[tail_s] = word_count
            word_embedding_matrix[word_count, :] = np.random.uniform(low=-0.1, high=0.1, size=word_dim)
        relation_num = relation_idx_dict.get(relation_str, 0)
        word_list = sentence.split()[:-1]
        sentence_idx = [word_idx_dict.get(w, 1) for w in word_list]
        lefnum = 0
        rignum = 0
        for i, w in enumerate(word_list):
            if w == head_s:
                lefnum = i
            if w == tail_s:
                rignum = i
        positions = [lefnum, rignum]
        epos = sorted([lefnum, rignum])
        if not bags_train.get('%s\t%s\t%s' % (e1, e2, relation_str), ''):
            bags_train['%s\t%s\t%s' % (e1, e2, relation_str)] = InstanceBag(entity_pair=entity_pair, \
                relation=relation_num, instance_count=1, sentences=[sentence_idx], \
                positions=[positions], entity_pos=[epos])
        else:
            bags_train['%s\t%s\t%s' % (e1, e2, relation_str)].instance_count += 1
            bags_train['%s\t%s\t%s' % (e1, e2, relation_str)].sentences.append(sentence_idx)
            bags_train['%s\t%s\t%s' % (e1, e2, relation_str)].positions.append(positions)
            bags_train['%s\t%s\t%s' % (e1, e2, relation_str)].entity_pos.append(epos)
    fp.close()
    word_count = max(list(word_idx_dict.values()))-1
    word_embedding_matrix = word_embedding_matrix[:word_count+2, :]

    fp = open(test_filename, 'r')
    while True:
        line = fp.readline().strip()
        if not line:
            break
        e1, e2, head_s, tail_s, relation_str, sentence, _ = line.split('\t')
        entity_pair = [head_s, tail_s]
        relation_num = relation_idx_dict.get(relation_str, 0)
        word_list = sentence.split()[:-1]
        sentence_idx = [word_idx_dict.get(w, 1) for w in word_list]
        lefnum = 0
        rignum = 0
        for i, w in enumerate(word_list):
            if w == head_s:
                lefnum = i
            if w == tail_s:
                rignum = i
        positions = [lefnum, rignum]
        epos = sorted([lefnum, rignum])
        if not bags_test.get('%s\t%s\t%s' % (e1, e2, relation_str), ''):
            bags_test['%s\t%s\t%s' % (e1, e2, relation_str)] = InstanceBag(entity_pair=entity_pair, \
            relation=relation_num, instance_count=1, sentences=[sentence_idx], \
            positions=[positions], entity_pos=[epos])
        else:
            bags_test['%s\t%s\t%s' % (e1, e2, relation_str)].instance_count += 1
            bags_test['%s\t%s\t%s' % (e1, e2, relation_str)].sentences.append(sentence_idx)
            bags_test['%s\t%s\t%s' % (e1, e2, relation_str)].positions.append(positions)
            bags_test['%s\t%s\t%s' % (e1, e2, relation_str)].entity_pos.append(epos)
    fp.close()

def cut_and_pad(sentence, entity_pos, filter_h=3, max_len=80):
    x = []
    if entity_pos[0] == entity_pos[1]:
        if (entity_pos[1] + 1) < len(sentence):
            entity_pos[1] += 1
        else:
            entity_pos[0] -= 1
    if len(sentence) < max_len:
        x += sentence
    else:
        idx = range(entity_pos[0], entity_pos[1] + 1)
        if len(idx) > max_len:
            idx = idx[:max_len]
            x += [sentence[i] for i in idx]
            entity_pos[0] = 0
            entity_pos[1] = len(idx) - 1
        else:
            x += [sentence[i] for i in idx]
            before = entity_pos[0] - 1
            after = entity_pos[1] + 1
            entity_pos[0] = 0
            entity_pos[1] = len(idx) - 1
            num_added = 0
            while True:
                added = False
                if (before >= 0) and (len(x) < max_len):
                    x = [sentence[before]] + x
                    added = True
                    num_added += 1
                if (after < len(sentence)) and (len(x) < max_len):
                    x += [sentence[after]]
                    added = True
                if not added:
                    break
                before -= 1
                after += 1
            entity_pos[0] += num_added
            entity_pos[1] += num_added
    pad = int(filter_h / 2)
    x = [0] * pad + x
    while len(x) < max_len + 2 * pad:
        x.append(0)
    return x, entity_pos

def get_relative_position(sentence_length, entity_pos, filter_h=3, max_len=80):
    index = np.arange(min(sentence_length, max_len))
    pf1 = index - entity_pos[0] + 1 + 51
    pf2 = index - entity_pos[1] + 1 + 51
    pf1 = [max(1, min(p, 101)) for p in pf1]
    pf2 = [max(1, min(p, 101)) for p in pf2]
    pad = int(filter_h / 2)
    x1 = [0] * pad + pf1
    x2 = [0] * pad + pf2
    while len(x1) < max_len + 2 * pad:
        x1.append(0)
        x2.append(0)
    return [x1, x2]

def make_data(data, filter_h, max_len):
    new_data = []
    for bag in data:
        entity_pair, relation, instance_count, sentences, positions, entity_pos = bag.decompose()
        new_sentence = []
        new_position = []
        new_entity_pos = []
        for i, sentence in enumerate(sentences):
            tmp_sen, tmp_epos = cut_and_pad(sentence, entity_pos[i], filter_h=filter_h, max_len=max_len)
            new_sentence.append(tmp_sen)
            new_entity_pos.append(tmp_epos)
            tmp_pos = get_relative_position(len(sentence), positions[i], filter_h=filter_h, max_len=max_len)
            new_position.append(tmp_pos)
        new_instance = InstanceBag(entity_pair, relation, instance_count, new_sentence, new_position, new_entity_pos)
        new_data.append(new_instance)
    return new_data

def save_bags(bags, output_filename):
    fp = open(output_filename, 'w')
    fp.write('%d\n' % len(bags))
    for bag in bags:
        fp.write('%s %s\n' % (bag.entity_pair[0], bag.entity_pair[1]))
        fp.write('%d\n' % bag.relation)
        fp.write('%d\n' % bag.instance_count)
        for i in range(len(bag)):
            for j in range(len(bag.sentences[i])):
                fp.write('%d ' % bag.sentences[i][j])
            fp.write('\n')
        for i in range(len(bag)):
            for j in range(len(bag.positions[i][0])):
                fp.write('%d ' % bag.positions[i][0][j])
            fp.write('\n')
            for j in range(len(bag.positions[i][1])):
                fp.write('%d ' % bag.positions[i][1][j])
            fp.write('\n')
        for i in range(len(bag)):
            fp.write('%d %d\n' % (bag.entity_pos[i][0], bag.entity_pos[i][1]))
    fp.close()

def load_bags(input_filename):
    bags = []
    fp = open(input_filename, 'r')
    num_bags = int(fp.readline().strip())
    for i in range(num_bags):
        entity_pair = fp.readline().strip().split()
        relation = int(fp.readline().strip())
        instance_count = int(fp.readline().strip())
        sentences = []
        for j in range(instance_count):
            sentences.append([int(w) for w in fp.readline().strip().split()])
        positions = []
        for j in range(instance_count):
            x1 = [int(w) for w in fp.readline().strip().split()]
            x2 = [int(w) for w in fp.readline().strip().split()]
            positions.append([x1, x2])
        entity_pos = []
        for j in range(instance_count):
            entity_pos.append([int(w) for w in fp.readline().strip().split()])
        bags.append(InstanceBag(entity_pair=entity_pair, relation=relation, instance_count=instance_count,
            sentences=sentences, positions=positions, entity_pos=entity_pos))
    return bags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NYT10 dataset.")
    parser.add_argument("--data_path", nargs='?', default="../data/NYT10", help="Path to read NYT10 dataset.")
    parser.add_argument("--filter_h", type=int, default=3, help="Length of convolutional filter.")
    parser.add_argument("--max_len", type=int, default=80, help="Maximum sentence length.")
    parser.add_argument("--output_path", nargs='?', default="../data/processed", help="Path to output processed data.")
    args = parser.parse_args()
    arg_dict = vars(args)
    logger = Logger(None)
    length = max([len(arg) for arg in arg_dict.keys()])
    for arg, value in arg_dict.items():
        logger("%s | %s" % (arg.ljust(length).replace('_', ' '), str(value)))
    logger("Load data.")
    load_wordvector(os.path.join(args.data_path, 'vec.bin'))
    load_relation(os.path.join(args.data_path, 'relation2id.txt'))
    load_train_and_test_data(os.path.join(args.data_path, 'train.txt'), os.path.join(args.data_path, 'test.txt'))
    logger("Make data.")
    train_bags = make_data(list(bags_train.values()), args.filter_h, args.max_len)
    test_bags = make_data(list(bags_test.values()), args.filter_h, args.max_len)
    logger("Save data.")
    if args.data_path != args.output_path:
        if os.path.exists(args.output_path):
            os.system("rm -rf %s" % args.output_path)
        os.mkdir(args.output_path)
    np.save(os.path.join(args.output_path, 'word_vector.npy'), word_embedding_matrix)
    pickle.dump(word_idx_dict, open(os.path.join(args.output_path, 'dictionary.p'), 'wb'))
    save_bags(train_bags, os.path.join(args.output_path, 'train_bags.txt'))
    save_bags(test_bags, os.path.join(args.output_path, 'test_bags.txt'))
