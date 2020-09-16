import re, os, json, time
import sqlite3 as sql
import numpy as np
import utils
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
from ontologies import CamRest676Ontology, MultiwozOntology, KvretOntology
from vocab import Vocab
from config import global_config as cfg



class Dataset(object):
    def __init__(self):
        self.word_tokenize = nltk_word_tokenize
        self.wn = WordNetLemmatizer()

class CamRest676(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_path = cfg.dataset_path
        self.raw_data_path = cfg.raw_data
        self.otlg = CamRest676Ontology(cfg.ontology_path)
        self.requestable_slots = self.otlg.requestable_slots
        self.informable_slots = self.otlg.informable_slots

        self.data_path = os.path.join(self.dataset_path,
            # 'CamRest676_preprocessed_add_dontcare_330.json')
            # 'CamRest676_preprocessed_add_dontcare_41.json')
            'CamRest676_preprocessed_add_request_47.json')

        db_json_path = cfg.db.replace('.db', '.json')
        self.db_json = json.loads(open(db_json_path).read().lower())
        if not os.path.exists(cfg.db):
            self._db_construct(cfg.db)
        self.db = sql.connect(cfg.db).cursor()
        self.v_to_s = self.get_value_to_slot_mapping(cfg.ontology_path)


        self.db_vec_size = cfg.db_vec_size
        self.z_length = cfg.z_length


    def get_value_to_slot_mapping(self, save_dir):
        v_to_s = {}
        otlg_json = json.loads(open(save_dir).read().lower())
        for slot, values in otlg_json['informable'].items():
            for v in values:
                v_to_s[v] = slot
        return v_to_s

    def preprocess_data(self):
        print('Preprocessing data')
        raw_data = json.loads(open(self.raw_data_path).read().lower())
        db_data = self.db_json
        sw_ent, mw_ent = self._value_key_map(db_data)
        vocab = Vocab(cfg.vocab_size, self.otlg.special_tokens)
        # delexicalization
        dialogs = {}
        for dial_id, dial in enumerate(raw_data):
            dialogs[dial_id] = {}
            dialogs[dial_id]['goal'] = dial['goal']
            turns = []
            for turn in dial['dial']:
                turn_num = turn['turn']
                constraint = dict((slot, []) for slot in self.informable_slots)
                constraint_flat, user_request, sys_request = [], [], []
                for slot_values in turn['usr']['slu']:
                    if slot_values['act'] == 'inform':
                        slot, value = slot_values['slots'][0][0], slot_values['slots'][0][1]
                        slot = 'restaurant-' + slot
                        if slot != 'restaurant-slot' and value not in ['dontcare', 'none']:
                            constraint[slot].extend(self.word_tokenize(value))
                            constraint_flat.extend(self.word_tokenize(value))
                        if value == 'dontcare':
                            constraint[slot].extend(['dontcare'])
                            constraint_flat.extend(['dontcare'])
                    elif slot_values['act'] == 'request':
                        user_request.append('[value_%s]'%slot_values['slots'][0][1])
                            # constraint[slot].extend(['do', "n't", 'care'])
                if turn['sys']['da']:
                    for s in turn['sys']['da']:
                        s = ['price', 'range'] if s == 'pricerange' else [s]
                        if s == [["area, centre"]]:
                            s = ['area']
                        sys_request.extend(s)
                user = self.word_tokenize(turn['usr']['transcript'])
                resp = ' '.join(self.word_tokenize(turn['sys']['sent']))
                resp = self._replace_entity(resp, sw_ent, mw_ent, constraint_flat)
                resp = resp.replace('[value_phone].', '[value_phone] .').replace('ok.', 'ok .')
                resp = resp.split()
                # try:
                turns.append({
                    'turn': turn_num,
                    'user': ' '.join(user),
                    'response': ' '.join(resp),
                    'constraint': json.dumps(constraint),
                    'user_request': ' '.join(user_request),
                    'sys_request': ' '.join(sys_request),
                    'db_match': len(self.db_json_search(constraint)),
                })
                for word in user + resp:
                    vocab.add_word(word)
            dialogs[dial_id]['log'] = turns

        # save preprocessed data
        with open(self.data_path, 'w') as f:
            json.dump(dialogs, f, indent=2)

        # construct vocabulary
        vocab.construct()
        vocab.save_vocab(self.dataset_path + 'vocab')
        return dialogs

    def _replace_entity(self, text, single_word_ent, multi_word_ent, constraint):
        text = re.sub(r'[cC][., ]*[bB][., ]*\d[., ]*\d[., ]*\w[., ]*\w', '[value_postcode]', text)
        text = re.sub(r'\d{5}\s?\d{6}', '[value_phone]', text)
        constraint_str = ' '.join(constraint)

        for value, slot in multi_word_ent.items():
            if value in constraint_str:
                continue
            text = text.replace(value, '[value_%s]'%slot)

        tokens = text.split()
        for value, slot in single_word_ent.items():
            # if value == 'ask':
            #     continue
            if value in constraint_str or value not in tokens:
                continue
            tokens[tokens.index(value)] = '[value_%s]'%slot
            if 'moderately' in tokens:
                tokens[tokens.index('moderately')] = '[value_pricerange]'
            text = ' '.join(tokens)

        return text

    def _value_key_map(self, db_data):
        single_word_ent, multi_word_ent = {}, {}
        for db_entry in db_data:
            for slot, value in db_entry.items():
                value = ' '.join(self.word_tokenize(value))
                if slot in self.requestable_slots:
                    if len(value.split()) == 1:
                        single_word_ent[value] = slot
                    else:
                        multi_word_ent[value] = slot
        single_word_ent =OrderedDict(sorted(single_word_ent.items(), key=lambda x: -len(x[0])))
        multi_word_ent = OrderedDict(sorted(multi_word_ent.items(), key=lambda x: -len(x[0])))
        with open(os.path.join(self.dataset_path, 'value_dict.json'), 'w') as f:
            json.dump({'single': single_word_ent, 'multi': multi_word_ent}, f, indent=2)
        return single_word_ent, multi_word_ent

    def _db_construct(self, save_dir=None):
        all_slots = ['id', 'name', 'food', 'pricerange','area', 'phone', 'postcode', 'address',
                            'type', 'location']
        if save_dir is None:
            save_dir = 'data/CamRest676/CamRestDB.db'
        conn = sql.connect(save_dir)
        cur = conn.cursor()
        # create table
        exec_create = "CREATE TABLE restaurant("
        for slot in all_slots:
            exec_create += "%s TEXT ," % slot
        exec_create = exec_create[:-1] + ");"
        cur.execute(exec_create)

        for entry in self.db_json:
            slots = ",".join([s for s in all_slots])
            exec_insert = "INSERT INTO restaurant(" + slots+")\nVALUES ("
            for slot in all_slots:
                if entry.get(slot):
                    exec_insert += '"'+ str(entry[slot])+'",'
                else:
                    exec_insert += 'NULL,'
            exec_insert = exec_insert[:-1] + ')'
            cur.execute(exec_insert)
        conn.commit()
        conn.close()

    def db_search(self, constraints):
        match_results = []
        sql_query = "select * from restaurant where"
        for s,v in constraints.items():
            s = s.split('-')[1]
            if v in ['dontcare', "do n't care", 'any'] or v == []:
                continue
            sql_query += " %s='%s' and"%(s, ' '.join(v))
        match_results = self.db.execute(sql_query[:-3]).fetchall()
        # print(len(match_results))
        return match_results

    def db_json_search(self, constraints):
        match_results = []
        for entry in self.db_json:
            if 'food' not in entry:
                entry_values = entry['area'] + ' ' + entry['pricerange']
            else:
                entry_values = entry['area'] + ' ' + entry['food'] + ' ' + entry['pricerange']
            match = True
            for s,v in constraints.items():
                if v in ['dontcare', "do n't care", 'any']:
                    continue
                if ' '.join(v) not in entry_values:
                    match = False
                    break
            if match:
                match_results.append(entry)
        # print(len(match_results))
        return match_results

    def degree_vec_mapping(self, match_num):
        l = [0.] * self.db_vec_size
        l[min(self.db_vec_size - 1, match_num)] = 1.
        return l

    def pointer_back(self, degree):
        if 1 not in degree:
            return '-'
        i = degree.index(1)
        if i == len(degree-1):
            return '>%d'%(len(degree-2))
        else:
            return str(i)

    def get_db_degree(self, z_samples, vocab):
        """
        returns degree of database searching and it may be used to control further decoding.
        One hot vector, indicating the number of entries found: [0, 1, 2, 3, >=4]
        :param z_samples: tensor of [B, Tz*informable_slot_num]
        :return: an one-hot *numpy* db results vector and a list of match numbers
        """
        db_vec = []
        match = []

        for all_slots_idx in z_samples:
            all_slots_idx_tuple = all_slots_idx.split(self.z_length)
            # print(all_slots_idx_tuple)
            constraints = {}
            for sidx, slot in enumerate(self.informable_slots):
                constraints[slot] = []
                slot_idx = all_slots_idx_tuple[sidx]
                for widx in slot_idx:
                    w = vocab.decode(widx)
                    if w in ['<eos_b1>', '<eos_b2>','<eos_b3>','<eos_b>']:
                        break
                    if w not in constraints[slot]:
                        constraints[slot].append(w)
            match_num = len(self.db_json_search(constraints))
            db_vec.append(self.degree_vec_mapping(match_num))
            match.append(match_num)
        return np.array(db_vec), match


class Kvret(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_path = cfg.dataset_path
        self.raw_data_path = {
            'train': cfg.raw_train_data,
            'dev': cfg.raw_dev_data,
            'test': cfg.raw_test_data
        }
        self.otlg = KvretOntology(cfg.ontology_path)
        self.requestable_slots = self.otlg.requestable_slots
        self.informable_slots = self.otlg.informable_slots

        self.data_path = {
            'train': os.path.join(self.dataset_path, 'train_preprocessed.json'),
            'dev': os.path.join(self.dataset_path, 'dev_preprocessed.json'),
            'test': os.path.join(self.dataset_path, 'test_preprocessed.json')
        }

        self.entities = json.loads(open(cfg.ontology_path).read().lower())
        self.get_value_to_slot_mapping(self.entities)

    def _tokenize(self, sent):
        return ' '.join(self.word_tokenize(sent))

    def _lemmatize(self, sent):
        return ' '.join([self.wn.lemmatize(_) for _ in sent.split()])

    def get_value_to_slot_mapping(self, entity_data):
        self.entity_dict = {}
        self.abbr_dict = {}
        for k in entity_data:
            if type(entity_data[k][0]) is str:
                for entity in entity_data[k]:
                    entity = self._lemmatize(self._tokenize(entity))
                    self.entity_dict[entity] = k
                    if k in ['event','poi_type']:
                        self.entity_dict[entity.split()[0]] = k
                        self.abbr_dict[entity.split()[0]] = entity
            elif type(entity_data[k][0]) is dict:
                for entity_entry in entity_data[k]:
                    for entity_type, entity in entity_entry.items():
                        entity_type = 'poi_type' if entity_type == 'type' else entity_type
                        entity = self._lemmatize(self._tokenize(entity))
                        self.entity_dict[entity] = entity_type
                        if entity_type in ['event', 'poi_type']:
                            self.entity_dict[entity.split()[0]] = entity_type
                            self.abbr_dict[entity.split()[0]] = entity


    def _replace_entity(self, response, prev_user_input, intent):
        response = re.sub(r'\d+-?\d*fs?', '[value_temperature]', response)
        response = re.sub(r'\d+\s?miles?', '[value_distance]', response)
        response = re.sub(r'\d+\s\w+\s(dr)?(ct)?(rd)?(road)?(st)?(ave)?(way)?(pl)?\w*[.]?','[value_address]',response)
        response = self._lemmatize(self._tokenize(response))
        response = response.replace('[ ', '[').replace(' ]', ']')
        requestable = self.otlg.requestable_slots_dict
        for v, k in sorted(self.entity_dict.items(), key=lambda x: -len(x[0])):
            start_idx = response.find(v)
            if start_idx == -1 or k not in requestable[intent]:
                continue
            end_idx = start_idx + len(v)
            while end_idx < len(response) and response[end_idx] != ' ':
                end_idx += 1
            # test whether they are indeed the same word
            lm1, lm2 = v.replace('.','').replace(' ','').replace("'",''), \
                       response[start_idx:end_idx].replace('.','').replace(' ','').replace("'",'')
            if lm1 == lm2 and lm1 not in prev_user_input and v not in prev_user_input:
                response = utils.clean_replace(response, response[start_idx:end_idx], '[value_%s]'%k)
        return response


    def _clean_constraint_dict(self, constraint_dict, intent, prefer='short'):
        """
        clean the constraint dict so that every key is in "informable" and similar to one in provided entity dict.
        :param constraint_dict:
        :return:
        """
        for s in list(constraint_dict.keys()):
            if s not in self.otlg.informable_slots:
                constraint_dict.pop(s)

        invalid_key = []
        constraint_dict_new = {}
        for k in constraint_dict:
            v = self._lemmatize(self._tokenize(constraint_dict[k]))
            v = re.sub(r'(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), v)
            v = re.sub(r'(\d+)\s?(mile)s?', lambda x: x.group(1) + ' ' + x.group(2),v)
            constraint_dict_new[k] = v.strip()
            if v in self.entity_dict:
                if prefer == 'short':
                    constraint_dict_new[k] = v
                elif prefer == 'long':
                    constraint_dict_new[k] = self.abbr_dict.get(v, v)
                else:
                    raise ValueError('what is %s prefer, bro?' % prefer)
            elif v.split()[0] in self.entity_dict:
                if prefer == 'short':
                    constraint_dict_new[k] = v.split()[0]
                elif prefer == 'long':
                    constraint_dict_new[k] = self.abbr_dict.get(v.split()[0],v)
                else:
                    raise ValueError('what is %s prefer, bro?' % prefer)
            else:
                invalid_key.append(k)

        # if invalid_key:
        #     print('invalid key', invalid_key)
        for key in invalid_key:
            # print(constraint_dict_new[key])
            constraint_dict_new.pop(key)

        return constraint_dict_new


    def preprocess_data(self):
        """
        Somerrthing to note: We define requestable and informable slots as below in further experiments
        (including other baselines):
        :param raw_data:
        :param add_to_vocab:
        :param data_type:
        :return:
        """
        vocab = Vocab(cfg.vocab_size, self.otlg.special_tokens)
        for data_type in ['train', 'dev', 'test']:
            print('Preprocessing %s data'%data_type)
            raw_data =  json.loads(open(self.raw_data_path[data_type], 'r').read().lower())
            precessed_dialogs = {}
            state_dump = {}
            for dial_id, raw_dial in enumerate(raw_data):
                precessed_dialog = []
                prev_utter = ''
                single_turn = {}
                constraint_flat = []
                constraint_dict = {}
                intent = raw_dial['scenario']['task']['intent']
                if cfg.domain != 'all' and cfg.domain != intent:
                    if intent not in ['navigate','weather','schedule']:
                        raise ValueError('what is %s intent bro?' % intent)
                    else:
                        continue
                for turn_num,dial_turn in enumerate(raw_dial['dialogue']):
                    state_dump[(dial_id, turn_num)] = {}
                    if dial_turn['turn'] == 'driver':
                        u = self._lemmatize(self._tokenize(dial_turn['data']['utterance']))
                        u = re.sub(r'(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), u)
                        single_turn['user'] = u
                        prev_utter += u
                    elif dial_turn['turn'] == 'assistant':
                        s = dial_turn['data']['utterance']
                        # find entities and replace them
                        s = re.sub(r'(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), s)
                        s = self._replace_entity(s, prev_utter, intent)
                        single_turn['response'] = s

                        # get constraints
                        for s,v in dial_turn['data']['slots'].items():
                            constraint_dict[intent + '-' + s] = v

                        constraint_dict = self._clean_constraint_dict(constraint_dict, intent)
                        constraint_flat = list(constraint_dict.values())

                        single_turn['constraint'] = json.dumps(constraint_dict)
                        single_turn['turn_num'] = len(precessed_dialog)
                        single_turn['dial_id'] = dial_id

                        if 'user' in single_turn:
                            state_dump[(dial_id, len(precessed_dialog))]['constraint'] = constraint_dict
                            precessed_dialog.append(single_turn)
                        single_turn = {}

                for single_turn in precessed_dialog:
                    for word_token in constraint_flat + \
                            single_turn['user'].split() + single_turn['response'].split():
                        vocab.add_word(word_token)
                precessed_dialogs[dial_id] = precessed_dialog

            with open(self.data_path[data_type],'w') as f:
                json.dump(precessed_dialogs,f,indent=2)

        # construct vocabulary
        vocab.construct()
        vocab.save_vocab(self.dataset_path + 'vocab')

        return



class Multiwoz(Dataset):
    def __init__(self):
        super().__init__()
        self.raw_data_path = cfg.raw_data
        self.otlg = MultiwozOntology()
        self.requestable_slots = self.otlg.requestable_slots
        self.informable_slots = self.otlg.informable_slots

        self.db = MultiwozDB(cfg.db_paths, self.otlg)
        self.db_vec_size = cfg.db_vec_size

        self.z_length = cfg.z_length
        self.z_eos_tokens = list(self.otlg.z_eos_map.values())


    def get_db_degree(self, z_samples, domains, vocab):
        # z_samples: [B, slot_num * z_length]
        db_vec = []
        match = []

        for bidx, z_sample in enumerate(z_samples):
            constraints = self.bspn_to_constraint_dict(z_sample, vocab)
            match_num =self.db.get_match_num(constraints)

            if ' ' in domains[bidx]:
                domains[bidx] = domains[bidx].split()[0]
            db_vec.append(self.db.addDBPointer(domains[bidx], match_num[domains[bidx]]))
            match.append(str(match_num[domains[bidx]]))

        return np.array(db_vec), match

    def bspn_to_constraint_dict(self, z_sample, vocab):
        constraints = {}
        all_slots_idx_tuple = z_sample.split(self.z_length)
        for sidx, dom_slot in enumerate(self.informable_slots):
            dom, slot = dom_slot.split('-')
            if dom not in constraints:
                constraints[dom] = {}
            constraints[dom][slot] = []

            slot_idx = all_slots_idx_tuple[sidx]
            for widx in slot_idx:
                w = vocab.decode(widx)
                if w in self.z_eos_tokens:
                    break
                if w not in constraints[dom][slot]:
                    constraints[dom][slot].append(w)

            if not constraints[dom][slot]:
                del constraints[dom][slot]
            else:
                constraints[dom][slot] = ' '.join(constraints[dom][slot]).strip()
            if not constraints[dom]:
                del constraints[dom]
        return constraints


class MultiwozDB(object):
    def __init__(self, db_path, ontology):
        self.otlg = ontology
        with open(db_path, 'r') as f:
            self.dbs = json.loads(f.read().lower())


    def oneHotVector(self, domain, num):
        """Return number of available entities for particular domain."""
        vector = [0,0,0,0]
        if num == '':
            return vector
        if domain != 'train':
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num == 1:
                vector = [0, 1, 0, 0]
            elif num <=3:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        else:
            if num == 0:
                vector = [1, 0, 0, 0]
            elif num <= 5:
                vector = [0, 1, 0, 0]
            elif num <=10:
                vector = [0, 0, 1, 0]
            else:
                vector = [0, 0, 0, 1]
        return vector


    def addBookingPointer(self, turn_da):
        """Add information about availability of the booking option."""
        # Booking pointer
        # Do not consider booking two things in a single turn.
        vector = [0, 0]
        if turn_da.get('booking-nobook'):
            vector = [1, 0]
        if turn_da.get('booking-book') or turn_da.get('train-offerbooked'):
            vector = [0, 1]
        return vector


    def addDBPointer(self, domain, match_num, return_num=False):
        """Create database pointer for all related domains."""
        # if turn_domains is None:
        #     turn_domains = db_domains
        if domain in self.otlg.db_domains:
            vector = self.oneHotVector(domain, match_num)
        else:
            vector = [0, 0, 0 ,0]
        return vector


    def get_match_num(self, constraints, return_entry=False):
        """Create database pointer for all related domains."""
        match = {'general': ''}
        entry = {}
        # if turn_domains is None:
        #     turn_domains = db_domains
        for domain in self.otlg.all_domains:
            match[domain] = ''
            if domain in self.otlg.db_domains and constraints.get(domain):
                matched_ents = self.queryJsons(domain, constraints[domain])
                match[domain] = len(matched_ents)
                if return_entry :
                    entry[domain] = matched_ents
        if return_entry:
            return entry
        return match


    def pointerBack(self, vector, domain):
        # multi domain implementation
        # domnum = cfg.domain_num
        if domain.endswith(']'):
            domain = domain[1:-1]
        if domain != 'train':
            nummap = {
                0: '0',
                1: '1',
                2: '2-3',
                3: '>3'
            }
        else:
            nummap = {
                0: '0',
                1: '1-5',
                2: '6-10',
                3: '>10'
            }
        if vector[:4] == [0,0,0,0]:
            report = ''
        else:
            num = vector.index(1)
            report = domain+': '+nummap[num] + '; '

        if vector[-2] == 0 and vector[-1] == 1:
            report += 'booking: ok'
        if vector[-2] == 1 and vector[-1] == 0:
            report += 'booking: unable'

        return report


    def queryJsons(self, domain, constraints, exactly_match=True, return_name=False):
        """Returns the list of entities for a given domain
        based on the annotation of the belief state
        constraints: dict e.g. {'pricerange': 'cheap', 'area': 'west'}
        """
        # query the db
        # if domain == 'taxi':
        #     return [{'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']),
        #     'taxi_types': random.choice(self.dbs[domain]['taxi_types']),
        #     'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
        # if domain == 'police':
        #     return self.dbs['police']
        if domain == 'hospital':
            if constraints.get('department'):
                for entry in self.dbs['hospital']:
                    if entry.get('department') == constraints.get('department'):
                        return [entry]
            else:
                return []

        valid_cons = False
        for v in constraints.values():
            if v not in ["not mentioned", ""]:
                valid_cons = True
        if not valid_cons:
            return []

        match_result = []


        if 'name' in constraints:
            for db_ent in self.dbs[domain]:
                if 'name' in db_ent:
                    cons = constraints['name']
                    dbn = db_ent['name']
                    # if cons == dbn:
                    # use a relaxed search constraint when searching by names
                    if dbn.endswith(cons) or dbn.startswith(cons) or cons.endswith(dbn) or cons.startswith(dbn):
                        db_ent = db_ent if not return_name else db_ent['name']
                        match_result.append(db_ent)
                        return match_result

        for db_ent in self.dbs[domain]:
            match = True
            for s, v in constraints.items():
                if s == 'name':
                    continue
                if s in ['people', 'stay'] or(domain == 'hotel' and s == 'day') or \
                (domain == 'restaurant' and s in ['day', 'time']):
                    continue

                skip_case = {"don't care":1, "do n't care":1, "dont care":1, "not mentioned":1, "dontcare":1, "":1}
                if skip_case.get(v):
                    continue

                if s not in db_ent:
                    # logging.warning('Searching warning: slot %s not in %s db'%(s, domain))
                    match = False
                    break

                # v = 'guesthouse' if v == 'guest house' else v
                # v = 'swimmingpool' if v == 'swimming pool' else v
                v = 'yes' if v == 'free' else v

                if s in ['arrive', 'leave']:
                    try:
                        h,m = v.split(':')   # raise error if time value is not xx:xx format
                        v = int(h)*60+int(m)
                    except:
                        match = False
                        break
                    time = int(db_ent[s].split(':')[0])*60+int(db_ent[s].split(':')[1])
                    if s == 'arrive' and v>time:
                        match = False
                    if s == 'leave' and v<time:
                        match = False
                else:
                    if exactly_match and v != db_ent[s]:
                        match = False
                        break
                    elif v not in db_ent[s]:
                        match = False
                        break

            if match:
                match_result.append(db_ent)

        if not return_name:
            return match_result
        else:
            if domain == 'train':
                match_result = [e['id'] for e in match_result]
            else:
                match_result = [e['name'] for e in match_result]
            return match_result


if __name__ == "__main__":
    from config import global_config as cfg
    cfg.init_handler('camrest')
    dataset = CamRest676(cfg)
    c0 = {'food': [], 'pricerange': [], 'area':[]}
    c1 = {'food': ['asian', 'oriental'], 'pricerange': [], 'area':[]}
    c2 = {'food': [], 'pricerange': ['cheap'], 'area':['centre']}
    c3 = {'food': ['modern', 'european'], 'pricerange': ['cheap'], 'area':['centre']}
    bgt = time.time()
    for i in range(10000):
        # dataset.db_search(c0)
        dataset.db_search(c1)
        dataset.db_search(c2)
        dataset.db_search(c3)
    print('sql time: ', time.time()-bgt)
    bgt = time.time()
    for i in range(10000):
        # dataset.db_json_search(c0)
        dataset.db_json_search(c1)
        dataset.db_json_search(c2)
        dataset.db_json_search(c3)
    print('json time: ', time.time()-bgt)
    dataset.preprocess_data()