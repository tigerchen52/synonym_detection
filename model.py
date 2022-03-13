import numpy as np
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import torch.nn.functional as F
from registry import register
from functools import partial
registry = {}
register = partial(register, registry=registry)


@register('rnn')
class RNN(nn.Module):
    def __init__(self, vocab_size=21257, dim=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.rnn = nn.LSTM(input_size=dim, hidden_size=int(dim // 2), batch_first=True, num_layers=2, bidirectional=True)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activiation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        x_embed = self.embedding(x)
        shape = x_embed.size()

        mask = mask.squeeze().cpu().detach().numpy()
        mask = [np.sum(e != 0) for e in mask]

        # rnn pack
        packed = pack_padded_sequence(x_embed, mask, batch_first=True, enforce_sorted=False)
        encoder_outputs_packed, (h_last, c_last) = self.rnn(packed)
        rnn_output, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)

        #rnn_output2, h_n = self.rnn(x_embed)
        output = list()
        for index in range(len(mask)):
            temp = rnn_output[index, mask[index]-1, :]
            output.append(temp)
        output = torch.reshape(torch.cat(output, dim=0), (shape[0], 300))

        output = self.dropout(output)
        output = self.linear1(output)
        output = self.activiation(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output


@register('cnn')
class CNN(nn.Module):
    def __init__(self, vocab_size=21257, dim=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.embedding.weight.data.uniform_(-1, 1)
        self.windows = [3, 4, 5]
        self.dropout = 0.1
        self.convs = list()
        for window in self.windows:
            self.convs.append(nn.Conv2d(1, dim, (window, 300)).cuda())
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.proj = nn.Linear(len(self.windows)*dim, 300)

    def forward(self, x, x_lens):
        embeddings = self.embedding(x)
        embeddings = embeddings.unsqueeze(1)

        poolings = list()
        for conv in self.convs:
            conv_f = conv(embeddings)
            conv_f = F.relu(conv_f.squeeze(3))
            pooling = conv_f.max(dim=-1)[0]
            poolings.append(pooling)
        poolings = self.dropout(torch.cat(poolings, dim=-1))
        new_embed = self.proj(poolings)
        return new_embed


@register('attention')
class Attention(nn.Module):
    def __init__(self, vocab_size=21257, dim=300):
        super(Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.encoders = nn.ModuleList([OriAttention() for _ in range(1)])
        self.sublayer = SublayerConnection(0.1, dim)

    def forward(self, x, mask):
        x = self.embedding(x)
        for i, encoder in enumerate(self.encoders):
            x = self.sublayer(x, lambda x: encoder(x, mask))
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]


@register('sum')
class SUM(nn.Module):
    def __init__(self, vocab_size=21257, dim=300):
        super(SUM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.embedding.weight.requires_grad = True

    def forward(self, x, mask):
        x = self.embedding(x)
        return x.masked_fill_(~mask, 0).sum(dim=1)


class OriAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = 1
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask):
        q = x
        k = x
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2)
                   for a in [q, k, v])

        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)
        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)
        return v_


def l2norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)


@register('pam')
class PAMEncoder(nn.Module):
    def __init__(self, vocab_size=21257, dim=300):
        super(PAMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.encoders = nn.ModuleList([PAMformer() for _ in range(1)])
        self.sublayer = SublayerConnection(0.1, dim)
        self.head = 1

    def forward(self, x, mask):
        x = self.embedding(x)

        shape = list(x.size())
        position = PositionalEncoding(shape[-1], shape[-2])
        pos_att = position(x)

        for i, encoder in enumerate(self.encoders):
            x = self.sublayer(x, lambda x: encoder(x, mask, pos_att))

        x = x.masked_fill_(~mask, 0).sum(dim=1)
        return l2norm(x)


class PAMformer(nn.Module):
    def __init__(self, dim=300):
        super().__init__()
        self.self_attention = SAM()
        self.pos_attention = PAM()
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask, position):
        pos = self.pos_attention(x, mask, position)
        self_att = self.self_attention(pos, mask)
        c = self.projection(self_att)
        return self.dropout(c)


class PAM(nn.Module):
    def __init__(self, dim=300):
        super().__init__()
        self.head = 1
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask, pos):
        q = pos
        k = pos
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2) for a in [q, k, v])
        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)

        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)
        v_ = self.projection(v_)
        return v_


class PAM2(nn.Module):
    def __init__(self, dim=300):
        super().__init__()
        self.head = 1
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask, pos):
        v = x
        pos.masked_fill_(~mask, 0.)
        v_ = torch.matmul(pos.transpose(1, 2), v)
        v_ = self.projection(v_)
        return v_


class SAM(nn.Module):
    def __init__(self, dim=300):
        super().__init__()
        att_dim = 32
        self.head = 1
        self.projectionq = nn.Sequential(
            nn.Linear(dim, att_dim),
            nn.ReLU()
        )
        self.projectionk = nn.Sequential(
            nn.Linear(dim, att_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask):
        q = self.projectionq(x)
        k = self.projectionk(x)
        v = x
        q, k, v = (split_last(a, (self.head, -1)).transpose(1, 2)
                   for a in [q, k, v])

        scores = torch.matmul(q, k.transpose(2, 3)) / (k.size(-1) ** 0.25)
        mask = torch.matmul(mask.float(), mask.transpose(1, 2).float()).bool()
        mask = mask.unsqueeze(1)
        mask = mask.repeat([1, self.head, 1, 1])
        scores.masked_fill_(~mask, -1e7)
        scores = F.softmax(scores, dim=2)
        scores = scores.transpose(2, 3)
        v_ = torch.matmul(scores, v)
        v_ = v_.transpose(1, 2).contiguous()
        v_ = merge_last(v_, 2)

        return v_


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, dropout, dim):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(self.norm(sublayer(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def cal_fixed_pos_att(max_len, window_size):
    win = (window_size - 1) // 2
    weight = float(1 / window_size)
    attn_dict = dict()
    for sen_len in range(1, max_len+1):
        attn = np.eye(sen_len)
        if sen_len < window_size:
            attn_dict[sen_len] = attn
            continue
        for i in range(sen_len):
            attn[i, i-win:i+win+1] = weight
        attn[0, 0:win+1] = weight
        attn_dict[sen_len] = torch.FloatTensor(attn)

    return attn_dict


class PositionalAttCached(nn.Module):
    def __init__(self, d_model, pos_attns, max_len=5000):
        super(PositionalAttCached, self).__init__()
        # Compute the positional encodings once in log space.
        self.d_model = d_model
        self.pos_attns = pos_attns
        self.max_len = max_len

    def forward(self, x):
        shape = list(x.size())
        pos_attn = self.pos_attns[shape[1]]
        p_e = Variable(pos_attn, requires_grad=False).cuda()
        p_e = p_e.repeat([shape[0], 1, 1])
        return p_e


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        position = position * 1
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        shape = list(x.size())
        p_e = Variable(self.pe[:, :x.size(1)], requires_grad=False).cuda()
        p_e = p_e.repeat([shape[0], 1, 1])
        return p_e


