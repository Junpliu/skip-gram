import collections
import numpy as np

import math
import os
import random

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

data_index = 0
import torch


class Options(object):
    def __init__(self, filepath, vocab_size, window_size, neg_sample_cnt, language):
        with open(filepath, 'r', encoding='UTF-8') as f:
            words = f.read().split()  # 17005207 words
        words_idx_lst, words_counts = self.create_vocab(words, vocab_size)
        self.sampled_data = self.subsampling(words_idx_lst, words_counts)
        print('sampled_data len = {}\t'
              'initial data vocab size = {}\t'
              'vacab size = {}\t'.format(len(self.sampled_data), self.init_vocab_size, vocab_size))
        self.dict_path = 'word2idx.size' + str(vocab_size)
        if language == 'chinese':
            self.dict_path = self.dict_path[:8] + '_chinese' + self.dict_path[8:]
        torch.save({
            'word2idx': self.word2idx,
            'word_lst': words_idx_lst
        }, self.dict_path)
        self.sampling_table = self.create_sampling_table(words_counts)

    def create_vocab(self, words, vocab_size):
        words_cnt_tuple_lst = [('UNK', -1)]
        counter = collections.Counter(words)
        self.init_vocab_size = len(counter)
        words_cnt_tuple_lst.extend(counter.most_common(vocab_size - 1))
        # print(words_cnt_tuple_lst[:20])
        self.word2idx = {item[1][0]: item[0] for item in enumerate(words_cnt_tuple_lst)}
        words_idx_lst = []
        unknown_cnt = 0
        for word in words:
            if word in self.word2idx:
                words_idx_lst.append(self.word2idx[word])
            else:
                words_idx_lst.append(0)
                unknown_cnt += 1
        words_cnt_lst = [item[1] for item in words_cnt_tuple_lst]
        words_cnt_lst[0] = unknown_cnt
        return words_idx_lst, words_cnt_lst

    def subsampling(self, words_idx_lst, frequency):
        fre_np = np.array(frequency)
        fre_np = fre_np / fre_np.sum()
        # print('frequency = ', fre_np[:20])

        sampling_p = (np.sqrt(fre_np / 0.001) + 1) * 0.001 / fre_np
        # print('prob : ', sampling_p[:20])
        sampled_data = []
        # print('before sumple words_idx_lst: ', len(words_idx_lst), words_idx_lst[:20])
        for word in words_idx_lst:
            if random.random() < sampling_p[word]:
                sampled_data.append(word)
        return sampled_data

    def create_sampling_table(self, frequency):
        frequency75 = np.array(frequency) ** 0.75
        sum_words = sum(frequency75)
        ratio = frequency75 / sum_words
        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []
        for idx, cnt in enumerate(count):
            sample_table += [idx] * int(cnt)
        return np.array(sample_table, dtype=np.long)

    def generate_batch(self, window_size, batch_size, count):
        data = self.sampled_data
        global data_index
        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)

        if data_index + span > len(data):
            data_index = 0
            self.process = False
        buffer = data[data_index:data_index + span]
        pos_u = []
        pos_v = []

        for i in range(batch_size):
            data_index += 1
            context[i, :] = buffer[:window_size] + buffer[window_size + 1:]
            labels[i] = buffer[window_size]
            if data_index + span > len(data):
                buffer[:] = data[:span]
                data_index = 0
                self.process = False
            else:
                buffer = data[data_index:data_index + span]

            for j in range(span - 1):
                pos_u.append(labels[i])
                pos_v.append(context[i, j])
        neg_v = np.random.choice(self.sampling_table, size=(batch_size * 2 * window_size, count))
        return np.array(pos_u), np.array(pos_v), neg_v


import json, csv
from scipy.stats import spearmanr
import math


def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def scorefunction(embed):
    f = open('./checkpoints/vocab.txt')
    line = f.readline()
    vocab = []
    wordindex = dict()
    index = 0
    while line:
        word = line.strip().split()[0]
        wordindex[word] = index
        index = index + 1
        line = f.readline()
    f.close()
    ze = []
    with open('./wordsim353/combined.csv') as csvfile:
        filein = csv.reader(csvfile)
        index = 0
        consim = []
        humansim = []
        for eles in filein:
            if index == 0:
                index = 1
                continue
            if (eles[0] not in wordindex) or (eles[1] not in wordindex):
                continue

            word1 = int(wordindex[eles[0]])
            word2 = int(wordindex[eles[1]])
            humansim.append(float(eles[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            index = index + 1
            score = cosine_similarity(value1, value2)
            consim.append(score)

    cor1, pvalue1 = spearmanr(humansim, consim)

    if 1 == 1:
        lines = open('./rw/rw.txt', 'r').readlines()
        index = 0
        consim = []
        humansim = []
        for line in lines:
            eles = line.strip().split()
            if (eles[0] not in wordindex) or (eles[1] not in wordindex):
                continue
            word1 = int(wordindex[eles[0]])
            word2 = int(wordindex[eles[1]])
            humansim.append(float(eles[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            index = index + 1
            score = cosine_similarity(value1, value2)
            consim.append(score)

    cor2, pvalue2 = spearmanr(humansim, consim)

    return cor1, cor2


if __name__ == '__main__':
    import torch
    ch = torch.load('word2idx.dict')
    print('ch = ', ch.keys())
    input()
    word2idx = ch['word2idx']
    valid = 0
    invalid = 0
    with open('evaluation/syntactic_question/word_relationship.questions', 'r') as f:
        for line in f.read().split('\n'):
            items = line.split()
            if len(items) < 3:
                break
            if items[0] in word2idx and items[1] in word2idx and items[2] in word2idx:
                valid += 1
            else:
                invalid += 1
                print(items[0], items[1], items[2])
    print('valid = ', valid, 'invalid = ', invalid)