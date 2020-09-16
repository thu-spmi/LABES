import json, random, copy
import numpy as np

def toss_(p):
    return random.randint(0, 99) <= p

def nan(v):
    return np.isnan(np.sum(v.data.cpu().numpy()))

def write_dict(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, indent=2)

def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        # idx = s[sidx:].find(r)
        idx = s.find(r)
        if idx == -1:
            return s, -1
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    # source, replace, target = s, r, t
    # count = 0
    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
        # count += 1
        # print(s, sidx)
        # if count == 20:
        #     print(source, '\n', replace, '\n', target)
        #     quit()
    return s


def padSeqs(sequences, maxlen=None, truncated=False, fixed_length=False,
                     pad_method='post',  trunc_method='pre', dtype='int32', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    if maxlen is not None and fixed_length:
        maxlen = maxlen
    elif maxlen is not None and truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list/array was found')
            continue  # empty list/array was found
        if trunc_method == 'pre':
            trunc = s[-maxlen:]
        elif trunc_method == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % trunc_method)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if pad_method == 'post':
            x[idx, :len(trunc)] = trunc
        elif pad_method == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % pad_method)
    return x


def position_encoding_init(self, n_position, d_pos_vec):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
                             if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc

class BeamState:
    def __init__(self, score, last_hidden, decoded, length):
        """
        Beam state in beam decoding
        :param score: sum of log-probabilities
        :param last_hidden: last hidden
        :param decoded: list of *Variable[1*1]* of all decoded words
        :param length: current decoded sentence length
        """
        self.score = score
        self.last_hidden = last_hidden
        self.decoded = decoded
        self.length = length

    def update_clone(self, score_incre, last_hidden, decoded_t):
        decoded = copy.copy(self.decoded)
        decoded.append(decoded_t)
        clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
        return clone

resp = '<go_r> there is a car collision nearby , but there is a car collision nearby . <eos_r>'
r = 'car collision nearby'
t = '[value_traffic_info]'

clean_replace(resp, r, t)