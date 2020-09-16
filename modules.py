import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from config import global_config as cfg

def cuda_(var):
    return var.cuda() if cfg.cuda else var


def get_one_hot_input(input_t, v_dim=None):
    """
    word index sequence -> one hot sparse input
    :param x_input_np: [B, Tenc]
    :return: tensor: [B,Tenc, V]
    """
    def to_one_hot(y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).fill_(1e-10).scatter_(1, y_tensor, 1)   #1e-10
        return y_one_hot.view(*y.shape, -1)

    # input_t = torch.from_numpy(x_input_np).long()   #[B, T]
    input_t_onehot = to_one_hot(input_t, n_dims=v_dim)   #[B,T,V]
    input_t_onehot[:, :, 0] = 1e-10   #<pad> to zero
    # input_t_onehot = [cuda_(t.to_sparse()) for t in input_t_onehot]
    # return input_t_onehot
    return cuda_(input_t_onehot)


def get_sparse_input_efficient(x_input_np):
    ignore_index = [0]
    result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size), dtype=np.float32)
    result.fill(1e-10)
    for b in range(x_input_np.shape[0]):
        for t in range(x_input_np.shape[1]):
            if x_input_np[b][t] not in ignore_index:
                result[b][t][x_input_np[b][t]] = 1.0
    result = torch.from_numpy(result).float()
    return result


def shift(pz_proba):
    """[summary]
    :param pz_proba: [B,T,V]
    :returns: shifted pz_proba
    """
    first_input = cuda_(torch.zeros((pz_proba.size(0), 1, pz_proba.size(2)))).fill_(1e-12)
    pz_proba = torch.cat([first_input, pz_proba], dim=1)
    return pz_proba[:, :-1].detach()



def gumbel_softmax(logits, temperature):
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def ST_gumbel_softmax_sample(y):
    """
    ST-gumbel-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        :param hidden: tensor of size [n_layer, B, H]
        :param encoder_outputs: tensor of size [B,T, H]
        """
        attn_energies = self.score(hidden, encoder_outputs)   # [B,T,H]
        if mask is None:
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        else:
            attn_energies.masked_fill_(mask, -1e20)
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]

        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context  # [B,1, H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)   # [B,T,H]
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = self.v(energy).transpose(1,2)   # [B,1,T]
        return energy


class Encoder(nn.Module):
    def __init__(self, embedding, input_size, embed_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.embedding = embedding
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)

    def forward(self, input_seqs, hidden=None, input_type='index'):
        if input_type == 'index':
            embedded = self.embedding(input_seqs)
        elif input_type == 'embedding':
            embedded = input_seqs
        outputs, hidden = self.gru(embedded, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class DynamicEncoder(nn.Module):
    def __init__(self, embedding, input_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.embedding = embedding
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)

    def forward(self, input_seqs, input_lens, hidden=None, input_type='index'):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [B, T] (input_type=index) or
                                                          [B,T,E] (input_type=embedding)
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(0)
        if input_type == 'index':
            embedded = self.embedding(input_seqs)
        elif input_type == 'embedding':
            embedded = input_seqs

        sort_idx = np.argsort(-input_lens)
        # print(sort_idx)
        # print('tensor:', torch.LongTensor(np.argsort(sort_idx)))
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        outputs = outputs[unsort_idx].contiguous()
        hidden = hidden[unsort_idx].contiguous()
        return outputs, hidden