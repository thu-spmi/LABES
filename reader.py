"""

"""
import numpy as np
import os, csv, random, logging, json, pickle
from datasets import CamRest676, Kvret, Multiwoz
from config import global_config as cfg
from vocab import Vocab
from multiwoz_preprocess import DataPreprocessor


class _ReaderBase:

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = Vocab(cfg.vocab_size)
        self.result_file = ''
        if cfg.use_act_slot_decoder:
            self.act_order = ['av', 'as']
        else:
            self.act_order = ['av']

    def _construct(self, *args):
        """
        load data, construct vocab and store them in self.train/dev/test
        :param args:
        :return:
        """
        raise NotImplementedError('This is an abstract class, bro')


    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        # del_l = []
        # for k in turn_bucket:
        #     if k >=5: del_l.append(k)
        #     logging.debug("bucket %d instance %d" % (k,len(turn_bucket[k])))
        #for k in del_l:
        #    turn_bucket.pop(k)
        return turn_bucket

    def _turn_bucket_to_batch(self, data, batch_size):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if len(batch) > 0.5 * batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        return all_batches

    def _construct_batches(self, dial_data, set_name, batch_size):
        while True:
            turn_bucket = self._bucket_by_turn(dial_data)
            all_batches = []
            for k in turn_bucket:
                batches = self._turn_bucket_to_batch(turn_bucket[k], batch_size)
                all_batches += batches
            skip = True if set_name != 'train' else False
            _, sup_turn_num, total_turn_num = self._mark_batch_as_supervised(all_batches, skip)
            real_spv_prop = sup_turn_num / total_turn_num
            exp_spv_prop = cfg.spv_proportion / 100
            if set_name != 'train' or exp_spv_prop*0.9 <= real_spv_prop <= exp_spv_prop*1.1:
                if set_name == 'train':
                    logging.info('Exp spv proportion: {:.3f} True spv proportion: {:.3f}'.
                  format(exp_spv_prop, real_spv_prop))
                transposed_all_batches = []
                for batch in all_batches:
                    transposed_all_batches.append(self._transpose_batch(batch))
                return transposed_all_batches
            else:
                logging.info('Exp spv proportion: {:.3f} True spv proportion: {:.3f}'.
                  format(exp_spv_prop, real_spv_prop))
                logging.info('Spv proportion bias higher than 10%: relabeling training data')
                random.shuffle(dial_data)

    def _mark_batch_as_supervised(self, all_batches, skip=False):
        supervised_num = int(len(all_batches) * cfg.spv_proportion / 100)
        sup_turn, total_turn = 0, 0
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    if skip:
                        turn['supervised'] = True
                    else:
                        turn['supervised'] = i < supervised_num
                        # if not turn['supervised']:
                        #     turn['db_vec'] = [0.] * cfg.db_vec_size # unsupervised learning. DB degree should be unknown
                    if turn['supervised']:
                        sup_turn += 1
                    total_turn += 1
        return all_batches, sup_turn, total_turn

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if cfg.dataset == 'multiwoz' and key == 'db_vec':
                        turn_domain = turn_batch['dom'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs


    def save_result(self, write_mode, results, field, write_title=False, result_save_path=None):
        path = cfg.result_path if result_save_path is None else result_save_path
        with open(path, write_mode) as rf:
            if write_title:
                rf.write(write_title+'\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None

    def save_loss(self, train_loss, valid_loss, events, file_name='loss.csv'):
        path = os.path.join(cfg.exp_path, file_name)
        with open(path, 'w') as rf:
            writer = csv.writer(rf)
            writer.writerow(['epoch'] + list(range(len(train_loss))))
            for k in train_loss[0]:
                writer.writerow(['train_'+k] + [i[k] for i in train_loss])
            for k in valid_loss[0]:
                writer.writerow(['valid_'+k] + [i[k] for i in valid_loss])
            writer.writerow(['events'] + events)
        return None


    def wrap_result(self, result_dict, eos_syntax=None):
        """
        wrap generated results
        :param gen_z:
        :param gen_m:
        :param turn_batch: dict of [i_1,i_2,...,i_b] with keys
        :return:
        """
        results = []
        if eos_syntax is None:
            eos_syntax = self.otlg.eos_syntax
        decode_fn = self.vocab.sentence_decode

        if cfg.dataset == 'camrest':
            field = ['dial_id', 'turn', 'user', 'bspn_gen', 'bspn', 'aspn_gen', 'aspn', 'resp_gen', 'resp', 'db_gen','db_match']
        elif cfg.dataset == 'multiwoz':
            field = ['dial_id', 'turn', 'user', 'bspn_gen', 'bspn', 'aspn_gen', 'aspn', 'resp_gen', 'resp', 'dom_gen', 'dom',
                        'db_gen','db_match', 'db_vec']
        elif cfg.dataset == 'kvret':
            field = ['dial_id', 'turn', 'user', 'bspn_gen', 'bspn', 'resp_gen', 'resp', 'db_gen','db_match']
        results = []
        for dial_id, turns in result_dict.items():
            # entry = {'dial_id': dial_id, 'turn': len(turns)}
            # for prop in field[2:]:
            #     entry[prop] = ''
            # results.append(entry)
            for turn_no, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    if key == 'bspn':

                        constraint = {}
                        for si, sn in enumerate(self.otlg.informable_slots):
                            v = turn[key][sn]
                            constraint[sn] = decode_fn(v, eos=self.otlg.z_eos_map[sn]).strip()
                            if constraint[sn] == '':
                                del constraint[sn]
                        entry[key] = constraint
                        # entry[key] = decode_fn(turn[key]['food'], eos=eos_syntax[key])
                    elif key == 'bspn_gen':
                        constraint = {}
                        idx_list = decode_fn(turn[key], eos=None).split()
                        for si, sn in enumerate(self.otlg.informable_slots):
                            b, e = si * cfg.z_length, (si+1) * cfg.z_length
                            temp = []
                            for s in idx_list[b:e]:
                                if s == self.otlg.z_eos_map[sn] or s == '<eos_b>':
                                    break
                                # if s == 'moderately':
                                #     print('covert moderately to moderate')
                                #     s = 'moderate'
                                # if s not in temp and s in self.otlg.slot_value_mask[sn] and s != '<pad>' and s != 'the':
                                if s not in temp:   # delete repeated words
                                    temp.append(s)
                            if temp:
                                constraint[sn] = ' '.join(temp).strip()
                            # temp.append('|')
                        # entry[key] = ' '.join(temp[:-1]).strip()
                        entry[key] = constraint
                        # entry[key] = decode_fn(turn[key], eos=eos_syntax[key])
                    elif key == 'aspn':
                        temp = []
                        for sn in self.act_order:
                            temp += decode_fn(turn[key][sn], eos='<eos_%s>'%sn).split() + ['|']
                        entry[key] = ' '.join(temp[:-1]).strip()
                    elif key == 'aspn_gen':
                        if key not in turn:
                            entry[key] = ''
                        else:
                            temp = []
                            idx_list = decode_fn(turn[key], eos=None).split()
                            for si, sn in enumerate(self.act_order):
                                b, e = si * cfg.a_length, (si+1) * cfg.a_length
                                for s in idx_list[b:e]:
                                    if s == '<eos_%s>'%sn:
                                        break
                                    if s not in temp:   # delete repeated words
                                        temp.append(s)
                                temp.append('|')
                            entry[key] = ' '.join(temp[:-1]).strip()
                    else:
                        v = turn.get(key, '')
                        entry[key] = decode_fn(v, eos=eos_syntax[key]) if key in eos_syntax and v != '' else v
                results.append(entry)
        # print(results[0:3])
        return results, field

    def get_glove_matrix(self, glove_path, initial_embedding_np):
        """
        return a glove embedding matrix
        :param self:
        :param glove_file:
        :param initial_embedding_np:
        :return: np array of [V,E]
        """

        cnt = 0
        vec_array = initial_embedding_np
        old_avg = np.average(vec_array)
        old_std = np.std(vec_array)
        vec_array = vec_array.astype(np.float32)
        new_avg, new_std = 0, 0

        if 'glove' in glove_path:
            ef = open(glove_path, 'r', encoding='UTF-8')
            for line in ef.readlines():
                line = line.strip().split(' ')
                word, vec = line[0], line[1:]
                vec = np.array(vec, np.float32)
                if not self.vocab.has_word(word):
                    continue
                word_idx = self.vocab.encode(word)
                if word_idx <self.vocab.vocab_size:
                    cnt += 1
                    vec_array[word_idx] = vec
                    new_avg += np.average(vec)
                    new_std += np.std(vec)
            new_avg /= cnt
            new_std /= cnt
        else:
            ef = open(glove_path, 'rb')
            emb_mat = pickle.load(ef)
            for word, vec in emb_mat.items():
                vec = np.array(vec, np.float32)
                if not self.vocab.has_word(word.lower()):
                    continue
                word_idx = self.vocab.encode(word.lower())
                if word_idx <self.vocab.vocab_size:
                    cnt += 1
                    vec_array[word_idx] = vec
                    new_avg += np.average(vec)
                    new_std += np.std(vec)
        ef.close()
        logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (
            cnt, old_avg, new_avg, old_std, new_std))
        return vec_array

    def cons_dict_to_indicator(self, constraint):
        indicator = []
        for k in self.otlg.informable_slots:
            eos = self.otlg.z_eos_map[k]
            cons_1st_widx = constraint[k][0]
            if eos == self.vocab.decode(cons_1st_widx):
                indicator.append(-1)
            else:
                indicator.append(1)
        return indicator

    def cons_tensors_to_indicator(self, z_samples):
        indicators = []
        for z_sample in z_samples:
            indicator = []
            for si, sn in enumerate(self.otlg.informable_slots):
                eos = self.otlg.z_eos_map[sn]
                b = si * cfg.z_length
                if eos == self.vocab.decode(z_sample[b]):
                    indicator.append(-1)
                else:
                    indicator.append(1)
            indicators.append(indicator)
        return np.array(indicators)


class CamRest676Reader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self.dataset = CamRest676()
        self.db = self.dataset.db
        self.otlg = self.dataset.otlg
        # self.otlg = CamRest676Ontology(cfg.ontology_path)
        self._construct()
        self.slot_value_mask = self.otlg.covert_mask_words_to_idx(self.vocab)

        # print(self.slot_value_mask)

    def _construct(self):
        """
        construct encoded train, dev, test set.
        """
        vocab_path = cfg.dataset_path + 'vocab.word2idx.json'
        if not os.path.exists(self.dataset.data_path) or not os.path.exists(vocab_path):
            self.data = self.dataset.preprocess_data()
        else:
            self.data = json.loads(open(self.dataset.data_path, 'r', encoding='utf-8').read().lower())
        self.vocab.load_vocab(cfg.vocab_path)

        encoded_data = self._get_encoded_data(self.data)
        self.train, self.dev, self.test = self._split_data(encoded_data, cfg.split)
        random.shuffle(self.train)
        self.train_batch = self._construct_batches(self.train, 'train', cfg.batch_size)
        self.dev_batch = self._construct_batches(self.dev, 'dev', cfg.batch_size)
        self.test_batch = self._construct_batches(self.test, 'test', cfg.batch_size)
        self.batches = {'train': self.train_batch, 'dev': self.dev_batch, 'test': self.test_batch}


    def _get_encoded_data(self, data):
        encoded_data = []

        for dial_id, dial in data.items():
            encoded_dial = []
            prev_response = []

            for turn in dial['log']:
                user = self.vocab.sentence_encode(turn['user'].split() + ['<eos_u>'])
                response = self.vocab.sentence_encode(turn['response'].split() + ['<eos_r>'])
                constraint = json.loads(turn['constraint'])
                # cons = {'food': [], 'pricerange': [], 'area':[]}
                # for k,v in constraint.items():
                #     constraint[k] = self.vocab.sentence_encode(v + [self.otlg.z_eos_map[k]])
                    # cons['food'].extend(self.vocab.sentence_encode(v))
                    # constraint[k] = self.vocab.sentence_encode(v + ['<eos_b>'])
                # cons['food'].append(5)
                # cons['pricerange'].append(5)
                # cons['area'].append(5)
                cons = {}
                for k in self.otlg.informable_slots:
                    cons[k] = self.vocab.sentence_encode(constraint[k] + [self.otlg.z_eos_map[k]])
                db_vec = self.dataset.degree_vec_mapping(turn['db_match'])
                turn_num = turn['turn']
                sys_offer_value = self.vocab.sentence_encode(turn['user_request'].split() + ['<eos_av>'])
                sys_ask_slot = self.vocab.sentence_encode(turn['sys_request'].split() + ['<eos_as>'])
                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn': turn_num,
                    'user': prev_response + user,
                    'resp': response,
                    'bspn': cons,
                    'filling_vec': self.cons_dict_to_indicator(cons),
                    'aspn': {'av': sys_offer_value, 'as': sys_ask_slot},
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'db_vec': db_vec,
                    'db_match': turn['db_match']
                })
                prev_response = response
            encoded_data.append(encoded_dial)
        return encoded_data

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        total = sum(split)
        dev_thr = len(encoded_data) * split[0] // total
        test_thr = len(encoded_data) * (split[0] + split[1]) // total
        train, dev, test = encoded_data[:dev_thr], encoded_data[dev_thr:test_thr], encoded_data[test_thr:]
        return train, dev, test

    def save_result_report(self, results, ctr_save_path=None):
        ctr_save_path =  cfg.global_record_path if ctr_save_path is None else ctr_save_path
        write_title = False if os.path.exists(ctr_save_path) else True

        unsup_prop = 0 if cfg.skip_unsup else 100 - cfg.spv_proportion
        exp = cfg.eval_load_path.split('/')[2] if 'experiments/' in cfg.eval_load_path else cfg.eval_load_path
        res = {'exp': exp, 'labeled data %': cfg.spv_proportion, 'unlabeled data %': unsup_prop,
                   'bleu': results['bleu'], 'match': results['match'], 'joint_goal': results['joint_goal'], 'request f1': results['req_f1'],
                   'act value gen f1': results['value_pred_f1'], 'act slot pred f1': results['slot_pred_f1']}
        for s, accu in results['slot_accu'].items():
            res[s] = accu
        res.update({'db_acc': results['db_acc'], 'epoch_num': results['epoch_num']})
        res.update({'slot_accu': results['slot_accu'], 'slot - p/r/f1': results['slot-p/r/f1']})
        res.update({'act_verbose': results['act_verbose']})
        # if 'final_train_loss' in results:
        #     res.update({'train_sup_nll': results['final_train_loss'].get('sup_p_joint_nll', 0),
        #                         'valid_nll': results['best_valid_loss']['loss'],
        #                         'test_nll': results['test_loss']['loss'],
        #                         'train_sup_pm': results['final_train_loss'].get('sup_m_loss', 0),
        #                         'valid_pm': results['best_valid_loss']['m_loss'],
        #                         'test_pm': results['test_loss']['m_loss'],
        #                         'train_unsup_nll': results['final_train_loss'].get('unsup_p_joint_nll', 0),
        #                         'train_unsup_pm': results['final_train_loss'].get('unsup_m_loss', 0)})

        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class KvretReader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self.dataset = Kvret()
        self.otlg = self.dataset.otlg
        # self.otlg = CamRest676Ontology(cfg.ontology_path)
        self._construct()

        # print(self.slot_value_mask)

    def _construct(self):
        """
        construct encoded train, dev, test set.
        """
        self.data = {}
        vocab_path = cfg.dataset_path + 'vocab.word2idx.json'
        if not os.path.exists(self.dataset.data_path['train']) or not os.path.exists(vocab_path):
            self.dataset.preprocess_data()
        for d in ['train', 'dev', 'test']:
            self.data[d] =  json.loads(open(self.dataset.data_path[d], 'r', encoding='utf-8').read().lower())
        self.vocab.load_vocab(cfg.vocab_path)

        self.train, self.dev, self.test = self._get_encoded_data(self.data)

        random.shuffle(self.train)
        self.train_batch = self._construct_batches(self.train, 'train', cfg.batch_size)
        self.dev_batch = self._construct_batches(self.dev, 'dev', cfg.batch_size)
        self.test_batch = self._construct_batches(self.test, 'test', cfg.batch_size)
        self.batches = {'train': self.train_batch, 'dev': self.dev_batch, 'test': self.test_batch}


    def _get_encoded_data(self, data):
        train, dev, test = [], [], []
        data_mapping = {'train': train, 'dev': dev, 'test':test}

        for data_type in data:

            for dial_id, dial in data[data_type].items():
                encoded_dial = []
                prev_response = []

                for turn in dial:
                    user = self.vocab.sentence_encode(turn['user'].split() + ['<eos_u>'])
                    response = self.vocab.sentence_encode(turn['response'].split() + ['<eos_r>'])
                    constraint = json.loads(turn['constraint'])
                    cons = {}
                    for k in self.otlg.informable_slots:
                        if k in constraint:
                            cons[k] = self.vocab.sentence_encode(constraint[k].split() + [self.otlg.z_eos_map[k]])
                        else:
                            cons[k] = self.vocab.sentence_encode([self.otlg.z_eos_map[k]])
                    # final input
                    encoded_dial.append({
                        'dial_id': dial_id,
                        'turn':  turn['turn_num'],
                        'user': prev_response + user,
                        'resp': response,
                        'bspn': cons,
                        'filling_vec': self.cons_dict_to_indicator(cons),
                        'db_vec': np.array([0]),
                        'db_match': 0,
                    })
                    prev_response = response
                data_mapping[data_type].append(encoded_dial)
        return train, dev, test


    def save_result_report(self, results, ctr_save_path=None):
        ctr_save_path =  cfg.global_record_path if ctr_save_path is None else ctr_save_path
        write_title = False if os.path.exists(ctr_save_path) else True

        unsup_prop = 0 if cfg.skip_unsup else 100 - cfg.spv_proportion
        exp = cfg.eval_load_path.split('/')[2] if 'experiments/' in cfg.eval_load_path else cfg.eval_load_path
        res = {'exp': exp, 'labeled data %': cfg.spv_proportion, 'unlabeled data %': unsup_prop,
                   'bleu': results['bleu'], 'match': results['match'], 'joint_goal': results['joint_goal'], 'request f1': results['req_f1'],
                   }
        for s, accu in results['slot_accu'].items():
            res[s] = accu
        res.update({'db_acc': results['db_acc'], 'epoch_num': results['epoch_num']})
        res.update({'slot_accu': results['slot_accu'], 'slot - p/r/f1': results['slot-p/r/f1']})

        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class MultiwozReader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self._construct()
        # self.slot_value_mask = self.otlg.covert_mask_words_to_idx(self.vocab)

        # print(self.slot_value_mask)

    def _construct(self):
        """
        construct encoded train, dev, test set.
        """
        vocab_file = os.path.join(cfg.data_path, 'vocab.word2idx.json')
        if not os.path.exists(cfg.data_file) or not os.path.exists(vocab_file):
            print('Data preprocessing')
            DataPreprocessor(do_analysis=True)

        self.dataset = Multiwoz()
        self.db = self.dataset.db
        self.otlg = self.dataset.otlg

        self.data = json.loads(open(cfg.data_file, 'r', encoding='utf-8').read().lower())
        self.vocab.load_vocab(cfg.vocab_path)

        self.train, self.dev, self.test  = self._get_encoded_data(self.data)

        random.shuffle(self.train)
        self.train_batch = self._construct_batches(self.train, 'train', cfg.batch_size)
        self.dev_batch = self._construct_batches(self.dev, 'dev', cfg.batch_size)
        self.test_batch = self._construct_batches(self.test, 'test', cfg.batch_size)
        # self.test_batch = self._construct_batches(self.test[:50], 'test', 8)
        self.batches = {'train': self.train_batch, 'dev': self.dev_batch, 'test': self.test_batch}


    def _get_encoded_data(self, data):
        test_list = [l.strip().lower() for l in open(cfg.test_list, 'r').readlines()]
        dev_list = [l.strip().lower() for l in open(cfg.dev_list, 'r').readlines()]
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
            self.test_files[fn] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1
            self.dev_files[fn] = 1

        train, dev, test = [], [], []

        length = {}

        for dial_id, dial in data.items():
            encoded_dial = []
            prev_response = ''

            for turn in dial['log']:
                user = self.vocab.sentence_encode(turn['user'].split() + ['<eos_u>'])
                response = self.vocab.sentence_encode(turn['resp'].split() + ['<eos_r>'])
                constraint = json.loads(turn['constraint'][0])
                cons = {}
                for d_s in self.otlg.informable_slots:
                    dom, slot = d_s.split('-')
                    if dom in constraint and slot in constraint[dom]:
                        cons[d_s] = self.vocab.sentence_encode(constraint[dom][slot].split() + [self.otlg.z_eos_map[d_s]])
                    else:
                        cons[d_s] = self.vocab.sentence_encode([self.otlg.z_eos_map[d_s]])
                sys_offer_value = self.vocab.sentence_encode(turn['sys_inform'].split() + ['<eos_av>'])
                sys_ask_slot = self.vocab.sentence_encode(turn['sys_request'].split() + ['<eos_as>'])

                if turn['name_from_db']:
                    prev_response = prev_response.replace('[value_name]', turn['name_from_db'])
                if prev_response != '':
                    prev_response = self.vocab.sentence_encode(prev_response.split() + ['<eos_r>'])
                else:
                    prev_response = []

                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn': turn['turn_num'],
                    'user': prev_response + user,
                    # 'pv_resp': prev_response,
                    'resp': response,
                    'bspn': cons,
                    'filling_vec': self.cons_dict_to_indicator(cons),
                    'aspn': {'av': sys_offer_value, 'as': sys_ask_slot},
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'db_vec': [int(i) for i in turn['pointer'].split(',')],
                    'db_match': turn['match'],
                    'dom': turn['turn_domain']
                })
                prev_response = turn['resp']
                if  len(turn['user']) not in length:
                    length[len(turn['user'])] = 1
                else:
                    length[len(turn['user'])] += 1
            if dial_id in self.test_files:
                test.append(encoded_dial)
            elif dial_id in self.dev_files:
                dev.append(encoded_dial)
            else:
                train.append(encoded_dial)
        # length = dict(sorted(length.items(), key=lambda kv:kv[0], reverse=True))
        # small, large=0,0
        # for k,v in length.items():
        #     if k<80:
        #         small += v
        #     else:
        #         large += v
        # print(small, large)
        # quit()
        return train, dev, test

    def save_result_report(self, results, ctr_save_path=None):
        ctr_save_path =  cfg.global_record_path if ctr_save_path is None else ctr_save_path
        write_title = False if os.path.exists(ctr_save_path) else True

        unsup_prop = 0 if cfg.skip_unsup else 100 - cfg.spv_proportion
        exp = cfg.eval_load_path.split('/')[2] if 'experiments/' in cfg.eval_load_path else cfg.eval_load_path
        res = {'exp': exp, 'labeled data %': cfg.spv_proportion, 'unlabeled data %': unsup_prop,
                   'bleu': results['bleu'], 'match': results['match'], 'success': results['success'],
                   'joint_goal': results['joint_goal'],
                   # 'domain': results['dom'],
                   'act value gen f1': results['value_pred_f1'], 'act slot pred f1': results['slot_pred_f1']}
        for s, accu in results['slot_accu'].items():
            res[s] = accu
        res.update({'db_acc': results['db_acc'], 'epoch_num': results['epoch_num']})
        res.update({'slot_accu': results['slot_accu'], 'slot - p/r/f1': results['slot-p/r/f1']})
        res.update({'act_verbose': results['act_verbose']})

        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])





if __name__ == '__main__':
    cfg.init_handler('kvret')
    # reader = CamRest676Reader()
    KvretReader()