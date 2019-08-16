import model_def
import getData
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
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
    def __init__(self, filepath, vocab_size, embedding_dims, window_size, neg_sample_cnt, epoch_num, batch_size, lr, GPU_available):
        print('vocab_size',vocab_size,'\nembedding_dims', embedding_dims,'\nwindow_size',window_size,'\nneg_sample_cnt',neg_sample_cnt,'\nepoch_num',epoch_num,'\nbatch_size',batch_size,'\nlr',lr)
        self.epoch_num = epoch_num
        self.lr = lr
        self.GPU_available = GPU_available
        dataset = getData.dataset(filepath, vocab_size, window_size, neg_sample_cnt)
        self.model = model_def.skip_gram(vocab_size, embedding_dims)
        if GPU_available and torch.cuda.is_available():
            self.model.cuda()
            print('moving model to GPU!!!')
        self.trainloader = DataLoader(dataset, batch_size=batch_size)
    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        print('Start training! ')
        losses = AverageMeter()
        epoch_time = AverageMeter()
        end = time.time()
        for epoch in range(self.epoch_num):
            for idx, (u_idx, v_idx, neg_labels) in enumerate(self.trainloader):
                if self.GPU_available and torch.cuda.is_available():
                    u_idx = u_idx.cuda()
                    v_idx = v_idx.cuda()
                    neg_labels = neg_labels.cuda()
                optimizer.zero_grad()
                loss = self.model(u_idx, v_idx, neg_labels)
                losses.update(loss.item(), len(u_idx))
                loss.backward()
                optimizer.step()
            epoch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}]\t'
                  'Loss: {losses.avg:.2f}\t'
                  'Time: {epoch_time.avg}'.format(epoch, losses=losses, epoch_time=epoch_time))


if __name__ == '__main__':
    wc = word2vec(filepath='text8',
                 vocab_size=100000,
                 embedding_dims=200,
                 window_size=5,
                 neg_sample_cnt=10,
                 epoch_num=1000,
                 batch_size=16,
                 lr=0.2,
                 GPU_available=False)
    wc.train()