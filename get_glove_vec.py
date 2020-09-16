import utils, pickle, vocab
# glove_path = '/home/ouzj01/zhangyc/project/glove/glove.840B.300d.txt'
glove_path = 'data/glove/glove_ori.6B.50d.txt'
# save_path = 'data/glove/glove_multiwoz.840B.300d.txt'
# save_path = 'data/glove/glove_multiwoz.6B.50d.txt'
save_path = 'data/glove/glove_kvret.6B.50d.txt'

vocab = vocab.Vocab(1100)
vocab.load_vocab('data/kvret/vocab')
# vocab.load_vocab('data/MultiWOZ/processed/vocab')
vec_array = []
with open(glove_path, 'r', encoding='UTF-8') as ef:
    with open(save_path, 'w') as f:

        for line in ef.readlines():
            line_sep = line.strip().split(' ')
            word, vec = line_sep[0], line_sep[1:]
            if vocab.has_word(word):
                f.write(line)
ef.close()
