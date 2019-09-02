import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from scipy.stats import pearsonr
import numpy as np
from getData import Options
from model import skipgram


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class word2vec:
    def __init__(self, lr, checkpoint, inputfile, test_only, batch_size, language, vocabulary_size, optim, embedding_dim=200, epoch_num=20,
                 windows_size=5, neg_sample_num=10):
        self.inputfile = inputfile
        # print(inputfile.split('/'))
        # for item in inputfile.split('/'):
        #     self.inputfile = os.path.join(self.inputfile, item)
        self.embedding_dim = embedding_dim
        self.windows_size = windows_size
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.neg_sample_num = neg_sample_num
        self.checkpoint = checkpoint
        self.lr = lr
        self.language = language
        self.test_only = test_only
        self.optim = optim

        print('inputfile = ', self.inputfile,
              'language = ', self.language)

        self.word2idx = {}
        if test_only:
            self.model = skipgram(self.vocabulary_size, self.embedding_dim)
            print('test only, load checkpoint from ' + self.checkpoint)
            checkpoint = torch.load('checkpoints/' + self.checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])

    def train(self):
        print('processing data')
        self.op = Options(self.inputfile, self.vocabulary_size, self.windows_size, self.neg_sample_num, self.language, batch_size=self.batch_size)
        self.model = skipgram(self.vocabulary_size, self.embedding_dim)
        if self.language == 'chinese':
            word2idx_dict = torch.load('word2idx_chinese.dict')
        else:
            word2idx_dict = torch.load('word2idx.dict')
        self.word2idx = word2idx_dict['word2idx']

        print('start training')
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if self.optim == 'adam':
            print('adam optim')
            optimizer = optim.SparseAdam(self.model.parameters(), lr=self.lr)
        start_epoch = 0
        global best_p                # english word relatedness
        best_p = 0
        global best_loss
        best_loss = 1000000000
        start_batch = 0

        if len(self.checkpoint) != 0:
            print('load checkpoint from ' + self.checkpoint)
            checkpoint = torch.load('checkpoints/' + self.checkpoint)
            start_epoch = checkpoint['epoch']
            if 'best_p' in checkpoint.keys():
                best_p = checkpoint['best_p']
            else:
                print('pretrained model has not key \'best_p\'! ')
            if 'best_loss' in checkpoint.keys():
                best_loss = checkpoint['best_loss']
            else:
                print('pretrained model has not key \'best_loss\'! ')
            if 'batch_num' in checkpoint.keys():
                start_batch = checkpoint['batch_num']
                self.op.data_index = start_batch
            else:
                print('pretrained model has not key \'start_batch\'! ')
            print('epoch :{}, batch {}'.format(start_epoch, start_batch))
            self.model.load_state_dict(checkpoint['state_dict'])
            # try:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            # except:
            #     print('error loading state dict of optim')
        losses = AverageMeter()
        # start_epoch = 0
        print('start epoch ', start_epoch, ' start batch', start_batch)
        for epoch in range(start_epoch, self.epoch_num):
            start = time.time()
            self.op.process = True
            batch_num = 0
            if start_batch != 0:
                batch_num = start_batch
            batch_new = batch_num
            losses.reset()
            while self.op.process:
                # pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)
                pos_u, pos_v, neg_v = self.op.iter_batch(self.windows_size, self.batch_size, self.neg_sample_num)
                pos_u = torch.LongTensor(pos_u)
                pos_v = torch.LongTensor(pos_v)
                neg_v = torch.LongTensor(neg_v)

                optimizer.zero_grad()
                loss = self.model(pos_u, pos_v, neg_v, self.batch_size)
                losses.update(loss.item(), pos_u.shape[0])
                loss.backward()

                optimizer.step()

                # if batch_num % 30000 == 0:
                #     torch.save(model.state_dict(), './tmp/skipgram.epoch{}.batch{}'.format(epoch, batch_num))
                if batch_num % 5000 == 0:  #
                    end = time.time()
                    if self.language == 'english':
                        word_embeddings = self.model.input_embeddings()
                        p1, p2 = self.wordsim()
                        print('epoch,batch={:2d} {}:  pearsonr={:1.3f} {:1.3f}  words/sec = {:4.2f} batchs_time = {:.2f} loss={:4.3f}'
                              .format(epoch, batch_num, p1, p2, (batch_num - batch_new) * self.batch_size / (end - start),(end - start) / 60,
                                      losses.avg))
                    else:
                        print(
                            'epoch,batch={:2d} {}:  words/sec = {:4.2f} batchs_time = {:.2f} loss={:4.3f}'
                                .format(epoch, batch_num,
                                        (batch_num - batch_new) * self.batch_size / (end - start),
                                        (end - start) / 60,
                                        losses.avg))
                        if losses.avg < best_loss:
                            best_loss = losses.avg
                            torch.save({'state_dict': self.model.state_dict(),
                                        'best_loss': best_loss,
                                        'optimizer': optimizer.state_dict(),
                                        'epoch': epoch,
                                        'batch_num': batch_num + 1},
                                       './checkpoints/chinese/epoch{}_optim{}_lr{}_input{}'.format(epoch, self.optim, self.lr, self.inputfile[-9]))
                    batch_new = batch_num
                    start = time.time()
                batch_num = batch_num + 1
            if self.language == 'english':
                word_embeddings = self.model.input_embeddings()
                p1, p2 = self.wordsim()
                print('epoch {}\t'
                      'pearsonr: {:.6f} {:.6f} '.format(epoch, p1, p2))
                if p1 > best_p:
                    best_p = p1
                    torch.save({'state_dict': self.model.state_dict(),
                                'best_p': best_p,
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch + 1}, './checkpoints/epoch{}_optim{}_lr{}_input{}'.format(epoch, self.optim, self.lr, self.inputfile[-9]))
            else:
                print('epoch {}\t'
                      'loss: {:.6f} '.format(epoch, best_loss))
                if losses.avg < best_loss:
                    best_loss = losses.avg
                    torch.save({'state_dict': self.model.state_dict(),
                                'best_loss': best_loss,
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch + 1},
                               './checkpoints/chinese/epoch{}'.format(epoch, batch_num, self.lr))
        print("Optimization Finished!")

    def wordsim(self):
        case_cnt = 0
        rho_a = []
        rho_b = []
        with open(os.path.join('evaluation/word_relatedness_task/wordsim353', 'combined.tab'), 'r')as f:
            for i, line in enumerate(f.read().split('\n')[1:]):
                items = line.split()
                if len(items) == 0:
                    break
                if items[0] in self.word2idx and items[1] in self.word2idx:
                    case_cnt += 1
                    a = self.model.u_embeddings.weight.data[self.word2idx[items[0]]]
                    b = self.model.u_embeddings.weight.data[self.word2idx[items[1]]]
                    elem_wist = (a * b).sum().item()
                    a_mod = torch.sqrt((a * a).sum()).item()
                    b_mod = torch.sqrt((b * b).sum()).item()
                    rho_a.append((elem_wist / a_mod / b_mod))
                    rho_b.append((float(items[2])))
        return pearsonr(np.array(rho_a), np.array(rho_b))

    def syntactic(self):  # 0.1684105630293971
        def find_nearest_cosine(input):  # return the index of the nearest vector
            norm_input = input / np.sqrt(np.sum(input * input))
            # print('norm_a', norm_input)
            norm_embed = word_embeddings / np.repeat(np.sqrt(np.sum(word_embeddings * word_embeddings, axis=1)), self.embedding_dim).reshape(self.vocabulary_size, self.embedding_dim)
            # print('norm_b', norm_embed)
            cos_dist = np.sum(norm_input * norm_embed, axis=1)
            # print('cos_dict = ', cos_dist)
            return np.argmax(cos_dist)

        word_embeddings = self.model.input_embeddings()
        if self.test_only:
            ch = torch.load('word2idx.dict')
            word2idx = ch['word2idx']
        else:
            word2idx = self.word2idx
        valid = 0
        invalid = 0
        with open('evaluation/syntactic_question/word_relationship.questions', 'r') as f:
            question_lines = f.read().split('\n')
        with open('evaluation/syntactic_question/word_relationship.questions', 'r') as f:
            answers_lines = f.read().split('\n')
        cor_cnt = 0
        for i in range(len(question_lines)):
            ques_items = question_lines[i].split()
            ans_items = answers_lines[i].split()
            if ques_items[0] in word2idx and ques_items[1] in word2idx and ques_items[2] in word2idx:  # a - b = c - d
                valid += 1
                if valid % 500 == 0:
                    print('processing idx: ', valid, 'current accuracy = ', cor_cnt / valid)
                a = word_embeddings[self.word2idx[ques_items[0]]]
                b = word_embeddings[self.word2idx[ques_items[1]]]
                c = word_embeddings[self.word2idx[ques_items[2]]]
                # print(a[:10])
                # print(b[:10])
                # print(c[:10])
                # print((b + c - a)[:10])
                # input()
                output_idx = find_nearest_cosine(b + c - a)
                if output_idx == self.word2idx[ans_items[1]]:
                    cor_cnt += 1
            else:
                invalid += 1
                # print(ques_items[0], ques_items[1], ques_items[2])
        print('valid = ', valid, 'invalid = ', invalid)
        return cor_cnt / valid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--testonly', type=bool, default=False)

parser.add_argument('--language', type=str, default='chinese')              # chinese               english
parser.add_argument('--vocabsize', type=int, default=500000)                # 250000                500000
parser.add_argument('--inputfile', type=str, default='../../wiki_zh_main')  # '../../wiki_zh_test'  'text8'
args = parser.parse_args()

if __name__ == '__main__':
    print('adam optim not load optim') #
    if args.language != 'chinese' and args.language != 'english':
        input('error language option! ')
    wc = word2vec(inputfile=args.inputfile, checkpoint=args.checkpoint, test_only=args.testonly, language=args.language, lr=args.lr, batch_size=args.batch_size, vocabulary_size=args.vocabsize, optim=args.optim)
    if not args.testonly:
        wc.train()
    else:
        print('syntactic accuracy = ', wc.syntactic())
