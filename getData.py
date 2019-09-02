import collections
import numpy as np

import math
import os
import random

import torch
import gc
from numba import jit

class Options(object):
    def __init__(self, filepath, vocab_size, window_size, neg_sample_cnt, language, batch_size):
        with open(filepath, 'r', encoding='UTF-8') as f:
            words = f.read().split()  # 17005207 words
        self.dict_path = 'word2idx.size' + str(vocab_size)
        if language == 'chinese':
            self.dict_path = self.dict_path[:8] + '_chinese' + self.dict_path[8:]
        if os.path.exists(self.dict_path):
            print('loaded vacabulary')
            dict_file = torch.load(self.dict_path)
            word2idx = dict_file['word2idx']
            words_counts = dict_file['words_counts']
            words_idx_lst = []
            for word in words:
                words_idx_lst.append(word2idx.get(word, 0))
        else:
            words_idx_lst, words_counts, word2idx = self.create_vocab(words, vocab_size)
            torch.save({
                'word2idx': word2idx,
                'words_counts': words_counts
            }, self.dict_path)
        del words
        gc.collect()
        print('start release memory')
        del word2idx
        gc.collect()
        self.sampled_data = self.subsampling(words_idx_lst, words_counts)
        print('done sampling data')
        del words_idx_lst
        gc.collect()
        print('sampled_data len = {}\t'
              'initial data vocab size = {}\t'
              'vacab size = {}\t'.format(len(self.sampled_data), 0, vocab_size))
        if not os.path.exists(self.dict_path):
            print('sampled_data len = {}\t'
                  'initial data vocab size = {}\t'
                  'vacab size = {}\t'.format(len(self.sampled_data), 0, vocab_size))
        self.sampling_table = self.create_sampling_table(words_counts)
        self.data_len = len(self.sampled_data)
        self.data_index = 0
        self.context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        self.labels = np.ndarray(shape=(batch_size), dtype=np.int64)

    def create_vocab(self, words, vocab_size):
        words_cnt_tuple_lst = [('UNK', -1)]
        print('enter create')
        counter_all = collections.Counter(words)
        self.init_vocab_size = len(counter_all)
        counter_most = counter_all.most_common(vocab_size - 1)
        words_cnt_tuple_lst.extend(counter_most)
        print('extend done')
        word2idx = {item[1][0]: item[0] for item in enumerate(words_cnt_tuple_lst)}
        print('done first enumerate')
        words_idx_lst = []
        unknown_cnt = 0
        for word in words:
            if word in word2idx:
                words_idx_lst.append(word2idx[word])
            else:
                words_idx_lst.append(0)
                unknown_cnt += 1
        words_cnt_lst = [item[1] for item in words_cnt_tuple_lst]
        words_cnt_lst[0] = unknown_cnt
        print('return create vocab')
        return words_idx_lst, words_cnt_lst, word2idx

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
        # global data_index
        span = 2 * window_size + 1
        # context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        # labels = np.ndarray(shape=(batch_size), dtype=np.int64)

        if self.data_index + span > self.data_len:
            self.data_index = 0
            self.process = False
        buffer = data[self.data_index:self.data_index + span]
        pos_u = []
        pos_v = []

        for i in range(batch_size):
            self.data_index += 1
            self.context[i, :] = buffer[:window_size] + buffer[window_size + 1:]
            self.labels[i] = buffer[window_size]
            if self.data_index + span > self.data_len:
                buffer[:] = data[:span]
                self.data_index = 0
                self.process = False
            else:
                buffer = data[self.data_index:self.data_index + span]

            for j in range(span - 1):
                pos_u.append(self.labels[i])
                pos_v.append(self.context[i, j])
        neg_v = np.random.choice(self.sampling_table, size=(batch_size * 2 * window_size, count))
        return np.array(pos_u), np.array(pos_v), neg_v

    def iter_batch(self, window_size, batch_size, count):
        data = self.sampled_data
        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)

        pos_u = []
        pos_v = []
        end_idx = self.data_index + batch_size + span
        if end_idx >= self.data_len:
            self.data_index = 0
            end_idx = self.data_index + batch_size + span
            self.process = False
        buffer = list(data[self.data_index: end_idx])
        self.data_index = end_idx
        for i in range(batch_size):
            context[i, :] = buffer[i:i+window_size] + buffer[i+window_size+1:i+span]
            labels[i] = buffer[i+window_size]
            for j in range(span - 1):
                pos_v.append(context[i, j])
                pos_u.append(labels[i])
        neg_v = np.random.choice(self.sampling_table, size=(batch_size * 2 * window_size, count))
        return np.array(pos_u), np.array(pos_v), neg_v
#