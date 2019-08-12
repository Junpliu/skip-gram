import torch
import collections
import numpy as np
import random
class dataset():
    def __init__(self, filepath, vocab_size, window_size, neg_sample_cnt):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.neg_sample_cnt = neg_sample_cnt
        with open(filepath, 'r') as f:
            words = f.read().split()# 17005207 words
        print('former count of words in words lst:', len(set(words)))
        words_idx_lst, words_counts = self.create_vocab(words, vocab_size)
        for idx in range(10):
            print(words_counts[idx])
        print('all words lst len:', len(words_idx_lst))
        print('the size of the dictionary of words index lst:', len(set(words_idx_lst)))
        sampled_words_idx_lst = self.subsampling(words_idx_lst, words_counts)
        print('sampled words lst len:', len(sampled_words_idx_lst))
        print('the size of the new dictionary for sampled words lst:', len(set(sampled_words_idx_lst)))
        self.build_dataset(sampled_words_idx_lst)
        self.sampling_table = self.create_sampling_table(words_counts)

    def create_vocab(self, words, vocab_size):
        words_cnt_tuple_lst = [('UNK', -1)]
        words_cnt_tuple_lst.extend(collections.Counter(words).most_common(vocab_size - 1))
        word2idx = {item[1][0]: item[0] for item in enumerate(words_cnt_tuple_lst)}
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
        return words_idx_lst, words_cnt_lst
        
    def subsampling(self, words_idx_lst, frequency):
        fre_np = np.array(frequency)
        # numpy.array storing the prob of sub-sampling
        sampling_p = (np.sqrt(fre_np / 0.001) + 1) * 0.001 / fre_np
        sampled_data = []
        for word in words_idx_lst:
            if random.random() < sampling_p[word]:
                sampled_data.append(word)
        return sampled_data
    
    def build_dataset(self, words_idx):
        self.words_len = len(words_idx)
        # after calculation: the number of pairs is (2*len-1)*window_size-window_size*window_size
        self.dataitems_len = (2*self.words_len-1)*self.window_size-self.window_size*self.window_size
        print('self.dataitems_len = ', self.dataitems_len)
        self.trainingpairs = torch.LongTensor(self.dataitems_len, 2)
        cur_idx = 0
        flag = False
        for idx, word in enumerate(words_idx):
            if word >= 100000:
                print('error!')
                input('wait')
            if idx < self.window_size:
                span = idx + 1 + self.window_size
                start = 0
            elif idx + self.window_size >= self.words_len:
                span = self.window_size + self.words_len - idx
                start = idx - self.window_size
            else:
                span = 2 * self.window_size + 1
                start = idx - self.window_size
            for j in range(start, start + span):
                if j != idx:
                    self.trainingpairs[cur_idx][0] = words_idx[idx]
                    self.trainingpairs[cur_idx][1] = words_idx[j]
                    cur_idx += 1
        if cur_idx != self.dataitems_len:
            print('cur_idx error!', cur_idx)
            input()

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

    def __len__(self):# the count of words after sub-sampling
        return self.dataitems_len
    
    def __getitem__(self, idx):# torch.Tensor.size([batch_size * 2 * windows_size, 2])
        if idx >= self.dataitems_len:
            print('error!')
            input()
        return self.trainingpairs[idx][0], self.trainingpairs[idx][1], torch.from_numpy(np.random.choice(self.sampling_table, self.neg_sample_cnt)).long()
    
# ps: considering replacing the words having lower frequency with the same word 
#   'Unknown'. For example, set the size of dictionary to 100000, then there will
#   will be (253854 - 100000) words in former dictionary being replaced by 'Unknown'.