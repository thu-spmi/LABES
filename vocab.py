import logging, json
from collections import OrderedDict
import utils

class Vocab(object):
    def __init__(self, vocab_size, special_tokens=[]):
        self.vocab_size = vocab_size
        self._idx2word = {}
        self._word2idx = {}
        self._freq_dict = {}
        self.special_tokens = special_tokens
        for w in self.special_tokens:
            self._absolute_add_word(w)

    def __len__(self):
        return len(self._idx2word)

    def _absolute_add_word(self, w):
        idx = len(self)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual vocabulary set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in l:
            self._add_to_vocab(word)
            if len(self._idx2word) >= self.vocab_size:
                break

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path+'.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_true = len(self._idx2word)
        print('vocab file loaded from "'+vocab_path+'"')
        print('Vocabulary size: %d' % (self.vocab_size_true))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        utils.write_dict(vocab_path+'.word2idx.json', self._word2idx)
        utils.write_dict(vocab_path+'.freq.json', _freq_dict)

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def sentence_decode(self, index_list, eos=None):
        l = [self.decode(_) for _ in index_list]
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx])

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]

    def encode(self, word):
        word = '<unk>' if word not in self._word2idx else word
        return self._word2idx[word]

    def decode(self, idx):
        if type(idx) is not int:
            idx = int(idx.item())
        return self._idx2word.get(idx, '<unk>')