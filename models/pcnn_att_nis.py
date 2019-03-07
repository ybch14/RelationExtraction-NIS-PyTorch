#!/usr/bin/python3.6
#encoding=utf-8
#pytorch==0.4.1
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from utils import *
from preprocess import load_bags
from modules import Embedding, NIS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, \
        activation=nn.Tanh(), dropout=0.5):
        super(PCNN, self).__init__()
        self.out_channels = out_channels
        self.activation = activation
        self.window_size = kernel_size[0]
        self.dropout = nn.Dropout(dropout)
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels, 3, requires_grad=True))

    def forward(self, inputs, entity_pos):
        '''
        inputs: (bag_size, 1, seq_len, word_dim + 2 * position_dim)
        entity_pos: (bag_size, 2)
        '''
        # conv_output: (bag_size, out_channels, new_seq_len, 1)
        conv_output = self.cnn(inputs)
        activate_output = self.activation(conv_output)
        bag_size = inputs.shape[0]
        pool_list = []
        for i in range(bag_size):
            idx1 = int(entity_pos[i, 0]) + int(self.window_size / 2)
            idx2 = int(entity_pos[i, 1]) + int(self.window_size / 2)
            pool1, _ = torch.max(activate_output[i, :, :idx1, :], dim=1) # (out_channels, 1)
            pool2, _ = torch.max(activate_output[i, :, idx1:idx2, :], dim=1) # (out_channels, 1)
            pool3, _ = torch.max(activate_output[i, :, idx2:, :], dim=1) # (out_channels, 1)
            pool_output = torch.cat([pool1, pool2, pool3], dim=1) # (out_channels, 3)
            pool_list.append(pool_output.unsqueeze(0)) # (1, out_channels, 3)
        output = torch.cat(pool_list, dim=0) # (bag_size, out_channels, 3)
        output += self.bias.unsqueeze(0).expand(bag_size, self.out_channels, 3)
        output = self.activation(output)
        return output

class PCNN_ATT_NIS(nn.Module):
    def __init__(self, class_num, word_embedding_matrix, position1_embedding_matrix, \
        position2_embedding_matrix, filters, kernel_size, padding=0, activation=nn.Tanh(), \
        dropout=0.5, nis_hidden_dims=[]):
        super(PCNN_ATT_NIS, self).__init__()
        self.filters = filters
        self.embedding = Embedding(word_embedding_matrix, position1_embedding_matrix, position2_embedding_matrix)
        self.pcnn = PCNN(1, filters, kernel_size, padding=padding, activation=activation, dropout=dropout)
        self.nis = NIS(3 * filters, nis_hidden_dims)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(3 * filters, class_num, bias=True)

    def init_params(self):
        init.xavier_uniform_(self.pcnn.cnn.weight)
        init.constant_(self.pcnn.bias, 0)
        for i, param in enumerate(self.nis.linears.parameters()):
            if i % 2 == 0:
                init.xavier_uniform_(param)
            else:
                init.constant_(param, 0)
        init.xavier_uniform_(self.linear.weight)
        init.constant_(self.linear.bias, 0)

    def forward(self, sentence, position1, position2, entity_pos, label=None):
        word_seq = torch.LongTensor(sentence).to(device)
        pos1_seq = torch.LongTensor(position1).to(device)
        pos2_seq = torch.LongTensor(position2).to(device)
        epos_seq = torch.LongTensor(entity_pos).to(device)
        embedded_inputs = self.embedding(word_seq, pos1_seq, pos2_seq)
        pcnn_output = self.pcnn(embedded_inputs, epos_seq)
        pcnn_feature = pcnn_output.view(-1, 3 * self.filters)
        if label is not None: # train mode
            bag_size = pcnn_feature.shape[0]
            if bag_size == 1:
                dropout_output = self.dropout(pcnn_feature)
                output = self.linear(dropout_output)
            else:
                masked_feature = self.nis(pcnn_feature)
                masked_bag_size = masked_feature.shape[0]
                if masked_bag_size == 1:
                    dropout_output = self.dropout(masked_feature)
                    output = self.linear(dropout_output)
                else:
                    attention_feature = torch.mm(masked_feature, self.linear.weight.t())[:, label]
                    self.attention = F.softmax(attention_feature, dim=0)
                    weighted_masked_feature = torch.mul(masked_feature, self.attention.unsqueeze(-1))
                    bag_masked_feature = torch.sum(weighted_masked_feature, dim=0, keepdim=True)
                    dropout_output = self.dropout(bag_masked_feature)
                    output = self.linear(dropout_output)
            return output
        else: # test mode
            bag_size = pcnn_feature.shape[0]
            if bag_size == 1:
                dropout_output = self.dropout(pcnn_feature)
                bag_feature = self.linear(dropout_output).squeeze()
                bag_prob = F.softmax(bag_feature, dim=0).detach()
                return bag_prob
            else:
                masked_feature = self.nis(pcnn_feature)
                masked_bag_size = masked_feature.shape[0]
                if masked_bag_size == 1:
                    dropout_output = self.dropout(masked_feature)
                    bag_feature = self.linear(dropout_output).squeeze()
                    bag_prob = F.softmax(bag_feature, dim=0).detach()
                    return bag_prob
                else:
                    attention_feature = torch.mm(masked_feature, self.linear.weight.t())
                    attention_feature = F.softmax(attention_feature, dim=0)
                    max_prob = torch.zeros(self.class_num)
                    for i in range(self.class_num):
                        attention = attention_feature[:, i]
                        weighted_masked_feature = torch.mul(masked_feature, attention.unsqueeze(-1))
                        bag_masked_feature = torch.sum(weighted_masked_feature, dim=0, keepdim=True)
                        bag_masked_feature = self.dropout(bag_masked_feature)
                        bag_feature = self.linear(bag_masked_feature).squeeze()
                        bag_prob = F.softmax(bag_feature, dim=0).detach()
                        max_prob[i] = bag_prob[i]
                    return max_prob

def predict(model, relations, counts, sents, poss, eposs, class_num, seq_len):
    pos_num = class_num - 1
    num_bag = len(relations)
    all_prob = np.zeros(num_bag * pos_num, dtype=np.float32)
    one_hot = np.zeros(num_bag * pos_num, dtype=np.int32)
    y = np.array(relations, dtype=np.int32)
    with torch.no_grad():
        model.eval()
        for bag_idx, instance_relation in enumerate(relations):
            ins_count = counts[bag_idx]
            ins_sentence = np.array(sents[bag_idx], dtype=np.int32).reshape((ins_count, seq_len))
            ins_pos1 = np.array([poss[bag_idx][m][0] for m in range(ins_count)], dtype=np.int32).reshape((ins_count, seq_len))
            ins_pos2 = np.array([poss[bag_idx][m][1] for m in range(ins_count)], dtype=np.int32).reshape((ins_count, seq_len))
            ins_epos = np.array(eposs[bag_idx], dtype=np.int32).reshape((ins_count, 2))
            prob = model(ins_sentence, ins_pos1, ins_pos2, ins_epos)
            if torch.cuda.is_available():
                prob = prob.detach().cpu().numpy()
            else:
                prob = prob.detach().numpy()
            all_prob[(bag_idx * pos_num):((bag_idx+1) * pos_num)] = prob[1:]
            one_hot[(bag_idx * pos_num):((bag_idx+1) * pos_num)] = to_categorical(instance_relation, class_num).squeeze()[1:]
        model.train()
    return all_prob, one_hot, y

def train(snapshot_path, prfile_path, verbose=True, logger=None):
    parser = argparse.ArgumentParser(description="Train PCNN+ATT+NIS model.")
    parser.add_argument("--model", nargs='?', default="pcnn_att", help="Name of model.")
    parser.add_argument("--data_path", nargs='?', default="../data/processed", help="Path of input data.")
    parser.add_argument("--output_path", nargs='?', default="../results", help="Path of output log file, snapshot and prfile.")
    parser.add_argument("--filters", type=int, default=230, help="Number of convolutional filter channels.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Size of convolutional filter.")
    parser.add_argument("--class_num", type=int, default=53, help="Number of relations.")
    parser.add_argument("--seq_len", type=int, default=80, help="Length of sentences.")
    parser.add_argument("--batch_size", type=int, default=160, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--nis_hidden_dims", nargs="?", default="512, 256, 128, 64", help="Dimensions of NIS hidden layers.")
    args = parser.parse_args()
    arg_dict = vars(args)
    if verbose:
        logger = logger if logger is not None else print
        length = max([len(arg) for arg in arg_dict.keys()])
        for arg, value in arg_dict.items():
            logger("%s | %s" % (arg.ljust(length).replace('_', ' '), str(value)))
    logger("Init Variables.")
    nis_hidden_dims = [int(x) for x in args.nis_hidden_dims.split(',') if x]
    w = np.load(os.path.join(args.data_path, "word_vector.npy"))
    w[1:, :] = w[1:, :] / np.sqrt(np.sum(w[1:, :] * w[1:, :], axis=1)).reshape((-1, 1))
    word_embedding_matrix = torch.tensor(w, requires_grad=True)
    p1 = np.random.uniform(low=-1, high=1, size=[101, 5])
    p1 /= np.sqrt(np.sum(p1 * p1, axis=1)).reshape((-1, 1))
    position1_embedding_matrix = torch.tensor(np.vstack([np.zeros((1, 5)), p1]), requires_grad=True).float()
    p2 = np.random.uniform(low=-1, high=1, size=[101, 5])
    p2 /= np.sqrt(np.sum(p2 * p2, axis=1)).reshape((-1, 1))
    position2_embedding_matrix = torch.tensor(np.vstack([np.zeros((1, 5)), p2]), requires_grad=True).float()
    word_dim = word_embedding_matrix.shape[1]
    pos_dim = position1_embedding_matrix.shape[1]
    logger("Load train and test data.")
    train_bags = load_bags(os.path.join(args.data_path, "train_bags.txt"))
    test_bags = load_bags(os.path.join(args.data_path, "test_bags.txt"))
    logger("Load model.")
    model = PCNN_ATT_NIS(args.class_num, word_embedding_matrix, position1_embedding_matrix, position2_embedding_matrix, \
        args.filters, (args.kernel_size, word_dim + 2 * pos_dim), padding=(int(args.kernel_size/2), 0), \
        nis_hidden_dims=nis_hidden_dims).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/2), gamma=0.5)
    logger("Init model.")
    model.init_params()
    logger("Begin training.")
    for epoch in range(args.epochs):
        # train
        scheduler.step()
        np.random.shuffle(train_bags)
        train_batches = split_batch(train_bags, batch_size=args.batch_size)
        model.train()
        for i, batch in enumerate(train_batches):
            label = [bag.relation for bag in batch]
            pred_list = []
            for bag in batch:
                train_label = bag.relation
                train_sentence = np.array(bag.sentences, dtype=np.int32).reshape((bag.instance_count, args.seq_len+2*int(args.kernel_size/2)))
                train_position1 = np.array([bag.positions[m][0] for m in range(bag.instance_count)], dtype=np.int32).reshape((bag.instance_count, args.seq_len+2*int(args.kernel_size/2)))
                train_position2 = np.array([bag.positions[m][1] for m in range(bag.instance_count)], dtype=np.int32).reshape((bag.instance_count, args.seq_len+2*int(args.kernel_size/2)))
                train_entity_pos = np.array(bag.entity_pos, dtype=np.int32).reshape((bag.instance_count, 2))
                tmp_pred = model(train_sentence, train_position1, train_position2, train_entity_pos, label=train_label)
                pred_list.append(tmp_pred)
            pred = torch.cat(pred_list, dim=0)
            label = torch.LongTensor(label).to(device)
            loss = loss_function(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 300 == 299:
                logger("batch = %d / %d, loss = %f" % (i+1, len(train_batches), loss.item()))
        # test
        np.random.shuffle(test_bags)
        test_relations = [bag.relation for bag in test_bags]
        test_counts = [bag.instance_count for bag in test_bags]
        test_sents = [bag.sentences for bag in test_bags]
        test_poss = [bag.positions for bag in test_bags]
        test_eposs = [bag.entity_pos for bag in test_bags]
        prob, one_hot, y = predict(model, test_relations, test_counts, test_sents, test_poss, test_eposs, args.class_num, args.seq_len+2*int(args.kernel_size/2))
        test_p, test_r = eval_pr_ATT(prob, one_hot, y)
        logger("Epoch %d, test precision: %f; test recall: %f" % (epoch+1, test_p[-1], test_r[-1]))
        logger("Epoch %d, save pr and model." % (epoch+1))
        save_pr(prfile_path, epoch, test_p, test_r)
        torch.save(model.state_dict(), os.path.join(snapshot_path, "snapshot_%d.model" % epoch))