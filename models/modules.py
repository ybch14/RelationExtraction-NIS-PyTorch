#!/usr/bin/python3.6
#encoding=utf-8
#pytorch==0.4.1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    def __init__(self, word_vector, position1_vector, position2_vector):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_vector, freeze=False)
        self.position1_embedding = nn.Embedding.from_pretrained(position1_vector, freeze=False)
        self.position2_embedding = nn.Embedding.from_pretrained(position2_vector, freeze=False)

    def forward(self, word, position1, position2):
        '''
        word, position1, position2: (bag_size, seq_len)
        '''
        # embedded_word: (bag_size, seq_len, word_dim)
        embedded_word = self.word_embedding(word)
        # embedded_pos1/2: (bag_size, seq_len, position_dim)
        embedded_pos1 = self.position1_embedding(position1)
        embedded_pos2 = self.position2_embedding(position2)
        # output: (bag_size, 1, seq_len, word_dim + 2 * pos_dim)
        output = torch.cat([embedded_word, embedded_pos1, embedded_pos2], dim=2)
        output = output.unsqueeze(1)
        return output

class PCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, \
        activation=nn.Tanh(), dropout=0.5):
        super(PCNN, self).__init__()
        self.activation = activation
        self.window_size = kernel_size[0]
        self.dropout = nn.Dropout(dropout)
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)

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
        dropout_output = self.dropout(output)
        return dropout_output

class Attention(nn.Module):
    def __init__(self, in_channels, activation=nn.Tanh()):
        super(Attention, self).__init__()
        self.linear = nn.Linear(in_channels, 1, bias=True)
        self.activation = activation

    def forward(self, inputs, e1, e2):
        bag_size = inputs.shape[0]
        # v_relation: (bag_size, word_dim)
        v_relation = (e1 - e2).unsqueeze(0).expand(bag_size, -1)
        # att_concat: (bag_size, in_channels)
        att_concat = torch.cat([inputs, v_relation], dim=1)
        # activate_feature: (bag_size, in_channels)
        activate_feature = self.activation(att_concat)
        # omega: (bag_size)
        omega = self.linear(activate_feature).squeeze()
        # weight: (bag_size)
        weight = F.softmax(omega, dim=0)
        return weight

class NIS(nn.Module):
    def __init__(self, in_channels, hidden_dims=[100]):
        super(NIS, self).__init__()
        self.in_channels = in_channels
        dim_list = [in_channels] + hidden_dims + [1]
        self.linears = nn.ModuleList()
        def dense(in_channels, out_channels):
            return [nn.Linear(in_channels, out_channels, bias=True), nn.Sigmoid(), nn.Dropout(0.5)]
        for i in range(1, len(dim_list)):
            self.linears += dense(dim_list[i-1], dim_list[i])
        self.mask = None

    def forward(self, inputs):
        # inputs: (bag_size, dim)
        feature = inputs
        for layer in self.linears:
            feature = layer(feature)
        # self.mask, feature: (bag_size, 1)
        self.mask = torch.ge(feature, 0.5)
        # masked_output: (new_bag_size, 1)
        masked_output = torch.masked_select(inputs, self.mask).view(-1, inputs.shape[1])
        if masked_output.nelement() == 0:
            max_idx = int(torch.max(feature, dim=0)[1])
            masked_output = inputs[max_idx, :].unsqueeze(0)
        return masked_output