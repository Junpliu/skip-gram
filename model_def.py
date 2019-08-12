import torch
import torch.nn as nn
import torch.nn.functional as F
class skip_gram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(skip_gram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos, v_neg):
        print('batch_size', pos.shape(0))
        input('wait')
        embed_u = self.u_embeddings(pos[:, 0])
        embed_v = self.v_embeddings(pos[:, 1])

        score = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()# batch_size

        neg_embed_v = self.v_embeddings(v_neg)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target + sum_log_sampled
        return -1*loss.sum()/pos.shape(0)
