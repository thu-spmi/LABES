import os, json, copy, re
from collections import OrderedDict
from tqdm import tqdm
from nltk.tokenize import word_tokenize as tknz

from config import global_config as cfg
from ontologies import MultiwozOntology
from datasets import MultiwozDB
from vocab import Vocab

number_to_text = {
    '1': 'one', '2': 'two',  '3':'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8':'eight', '9': 'nine', '10': 'ten',
    '11': 'eleven', '12': 'twelve'
}


def clean_time(utter):
        utter = re.sub(r'(\d+) ([ap]\.?m)', lambda x: x.group(1) + x.group(2), utter)   # 9 am -> 9am
        utter = re.sub(r'((?<!\d)\d:\d+)(am)?', r'0\1', utter)
        utter = re.sub(r'((?<!\d)\d)am', r'0\1:00', utter)
        utter = re.sub(r'((?<!\d)\d)pm', lambda x: str(int(x.group(1))+12)+':00', utter)
        utter = re.sub(r'(\d+)(:\d+)pm', lambda x: str(int(x.group(1))+12)+x.group(2), utter)
        utter = re.sub(r'(\d+)a\.?m',r'\1', utter)
        return utter

def clean_text(text):

    text = text.strip()
    text = text.lower()
    text = text.replace(u"’", "'")
    text = text.replace(u"‘", "'")
    text = text.replace(';', ',')
    text = text.replace('"', ' ')
    text = text.replace('/', ' and ')
    text = text.replace("don't", "do n't")
    text = clean_time(text)
    baddata = { r'c\.b (\d), (\d) ([a-z])\.([a-z])': r'cb\1\2\3\4',
                        'c.b. 1 7 d.y': 'cb17dy',
                        'c.b.1 7 d.y': 'cb17dy',
                        'c.b 25, 9 a.q': 'cb259aq',
                        'isc.b 25, 9 a.q': 'is cb259aq',
                        'c.b2, 1 u.f': 'cb21uf',
                        'c.b 1,2 q.a':'cb12qa',
                        '0-122-336-5664': '01223365664',
                        'postcodecb21rs': 'postcode cb21rs',
                        r'i\.d': 'id',
                        ' i d ': 'id',
                        'Telephone:01223358966': 'Telephone: 01223358966',
                        'depature': 'departure',
                        'depearting': 'departing',
                        '-type': ' type',
                        # r"b[\s]?&[\s]?b": "bed and breakfast",
                        # "b and b": "bed and breakfast",
                        # r"guesthouse[s]?": "guest house",
                        # r"swimmingpool[s]?": "swimming pool",
                        # "wo n\'t": "will not",
                        # " \'d ": " would ",
                        # " \'m ": " am ",
                        # " \'re' ": " are ",
                        # " \'ll' ": " will ",
                        # " \'ve ": " have ",
                        r'(\d+),(\d+)': r'\1\2',
                        r'^\'': '',
                        r'\'$': '',
                                }
    for tmpl, good in baddata.items():
        text = re.sub(tmpl, good, text)

    text = re.sub(r'([a-zT]+)\.([a-z])', r'\1 . \2', text)   # 'abc.xyz' -> 'abc . xyz'
    text = re.sub(r'(\d+)\.\.? ', r'\1 . ', text)   # if 'abc. ' -> 'abc . '

    return text


class DataPreprocessor(object):
    def __init__(self, do_analysis=True):
        self.otlg = MultiwozOntology()

        self.data_path = 'data/MultiWOZ/MULTIWOZ2.1/data.json'
        self.save_path ='data/MultiWOZ/processed/'
        self.original_db_paths = {
            'attraction': 'data/MultiWOZ/MULTIWOZ2.1/attraction_db.json',
            'hospital': 'data/MultiWOZ/MULTIWOZ2.1/hospital_db.json',
            'hotel': 'data/MultiWOZ/MULTIWOZ2.1/hotel_db.json',
            'police': 'data/MultiWOZ/MULTIWOZ2.1/police_db.json',
            'restaurant': 'data/MultiWOZ/MULTIWOZ2.1/restaurant_db.json',
            'taxi': 'data/MultiWOZ/MULTIWOZ2.1/taxi_db.json',
            'train': 'data/MultiWOZ/MULTIWOZ2.1/train_db.json',
        }

        if do_analysis:
            self.analysis()
            self.normalize_multiwoz21_name()

        self.original_data = json.loads(open(self.data_path, 'r').read().lower())
        self.dialog_acts = json.loads(open('data/MultiWOZ/MULTIWOZ2.1/dialogue_acts.json', 'r').read().lower())

        self.db_path = self.save_path + 'db_processed.json'
        if not os.path.exists(self.db_path):
            self.preprocess_db(self.original_db_paths, self.otlg)
        self.db = MultiwozDB(self.db_path, self.otlg)

        self.delex_sg_valdict = json.loads(open(self.save_path + 'single_token_values.json', 'r').read())
        self.delex_mt_valdict = json.loads(open(self.save_path + 'multi_token_values.json', 'r').read())
        self.ambiguous_vals = json.loads(open(self.save_path + 'ambiguous_values.json', 'r').read())
        self.reference_nos = json.loads(open(self.save_path + 'reference_no.json', 'r').read())
        self.name_map = json.loads(open(self.save_path + 'name_map.json', 'r').read())
        self.otlg_values = json.loads(open(self.save_path + 'db_values.json', 'r').read())


        self.vocab = Vocab(cfg.vocab_size, self.otlg.special_tokens)

        self.preprocess_main()

    def analysis(self):
        compressed_raw_data = {}
        goal_of_dials = {}
        req_slots = {}
        info_slots = {}
        dom_count = {}
        dom_fnlist = {}
        all_domain_specific_slots = set()
        for domain in self.otlg.all_domains:
            req_slots[domain] = []
            info_slots[domain] = []

        data_jsonstr = open(self.data_path, 'r').read().lower()
        data = json.loads(data_jsonstr)
        ref_nos = list(set(re.findall(r'\"reference\"\: \"(\w+)\"', data_jsonstr)))


        for fn, dial in data.items():
            goals = dial['goal']
            logs = dial['log']

            # get compressed_raw_data and goal_of_dials
            compressed_raw_data[fn] = {'goal': {}, 'log': []}
            goal_of_dials[fn] = {}
            for dom, goal in goals.items():
                if dom != 'topic' and dom != 'message' and goal:
                    compressed_raw_data[fn]['goal'][dom] = goal
                    goal_of_dials[fn][dom] = goal

            for turn in logs:
                if not turn['metadata']:
                    compressed_raw_data[fn]['log'].append({'text': turn['text']})
                else:
                    meta = turn['metadata']
                    turn_dict = {'text': turn['text'], 'metadata': {}}
                    for dom, book_semi in meta.items():
                        book, semi = book_semi['book'], book_semi['semi']
                        record = False
                        for slot, value in book.items():
                            if value not in ['', []]:
                                record = True
                        if record:
                            turn_dict['metadata'][dom] = {}
                            turn_dict['metadata'][dom]['book'] = book
                        record = False
                        for slot, value in semi.items():
                            if value not in ['', []]:
                                record = True
                                break
                        if record:
                            for s, v in copy.deepcopy(semi).items():
                                if v == 'not mentioned':
                                    del semi[s]
                            if not turn_dict['metadata'].get(dom):
                                turn_dict['metadata'][dom] = {}
                            turn_dict['metadata'][dom]['semi'] = semi
                    compressed_raw_data[fn]['log'].append(turn_dict)


                # get domain statistics
                dial_type = 'multi' if 'mul' in fn or 'MUL' in fn else 'single'
                if fn in ['pmul2756.json', 'pmul4958.json', 'pmul3599.json']:
                    dial_type = 'single'
                dial_domains = [dom for dom in self.otlg.all_domains if goals[dom]]
                dom_str = ''
                for dom in dial_domains:
                    if not dom_count.get(dom+'_'+dial_type):
                        dom_count[dom+'_'+dial_type] = 1
                    else:
                        dom_count[dom+'_'+dial_type] += 1
                    if not dom_fnlist.get(dom+'_'+dial_type):
                        dom_fnlist[dom+'_'+dial_type] = [fn]
                    elif fn not in dom_fnlist[dom+'_'+dial_type]:
                        dom_fnlist[dom+'_'+dial_type].append(fn)
                    dom_str += '%s_'%dom
                dom_str = dom_str[:-1]
                if dial_type=='multi':
                    if not dom_count.get(dom_str):
                        dom_count[dom_str] = 1
                    else:
                        dom_count[dom_str] += 1
                    if not dom_fnlist.get(dom_str):
                        dom_fnlist[dom_str] = [fn]
                    elif fn not in dom_fnlist[dom_str]:
                        dom_fnlist[dom_str].append(fn)
                ######

                # get informable and requestable slots statistics
                for domain in self.otlg.all_domains:
                    info_ss = goals[domain].get('info', {})
                    book_ss = goals[domain].get('book', {})
                    req_ss = goals[domain].get('reqt', {})
                    for info_s in info_ss:
                        all_domain_specific_slots.add(domain+'-'+info_s)
                        if info_s not in info_slots[domain]:
                            info_slots[domain]+= [info_s]
                    for book_s in book_ss:
                        if 'book_' + book_s not in info_slots[domain] and book_s not in ['invalid', 'pre_invalid']:
                            all_domain_specific_slots.add(domain+'-'+book_s)
                            info_slots[domain]+= ['book_' + book_s]
                    for req_s in req_ss:
                        if req_s not in req_slots[domain]:
                            req_slots[domain]+= [req_s]



        # result statistics
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        with open(self.save_path+'req_slots.json', 'w', encoding='utf-8') as sf:
            json.dump(req_slots,sf,indent=2)
        with open(self.save_path+'info_slots.json', 'w', encoding='utf-8') as sf:
            json.dump(info_slots,sf,indent=2)
        with open(self.save_path+'all_domain_specific_info_slots.json', 'w', encoding='utf-8') as sf:
            json.dump(list(all_domain_specific_slots),sf,indent=2)
        with open(self.save_path+'goal_of_each_dials.json', 'w', encoding='utf-8') as sf:
            json.dump(goal_of_dials, sf, indent=2)
        with open(self.save_path+'compressed_data.json', 'w', encoding='utf-8') as sf:
            json.dump(compressed_raw_data, sf, indent=2)
        with open(self.save_path + 'domain_count.json', 'w', encoding='utf-8') as sf:
            single_count = [d for d in dom_count.items() if 'single' in d[0]]
            multi_count = [d for d in dom_count.items() if 'multi' in d[0]]
            other_count = [d for d in dom_count.items() if 'multi' not in d[0] and 'single' not in d[0]]
            dom_count_od = OrderedDict(single_count+multi_count+other_count)
            json.dump(dom_count_od, sf, indent=2)
        with open(self.save_path + 'reference_no.json', 'w', encoding='utf-8') as sf:
            json.dump(ref_nos,sf,indent=2)
        with open(self.save_path + 'domain_files.json', 'w', encoding='utf-8') as sf:
            json.dump(dom_fnlist, sf, indent=2)


    def normalize_multiwoz21_name(self):
        data20 = 'data/MultiWOZ/compressed_data_2.0.json'
        data21 = self.save_path + 'compressed_data.json'
        data20 = json.loads(open(data20, 'r').read().lower())
        data21 = json.loads(open(data21, 'r').read().lower())

        name_map = {}

        db = {}
        for dom, db_path in self.original_db_paths.items():
            if dom not in self.otlg.db_domains:
                continue
            with open(db_path, 'r') as f:
                db[dom] = json.loads(f.read().lower())

        for fn, dial in data21.items():
            logs = dial['log']

            for tidx, turn in enumerate(logs):
                if 'metadata' in turn:
                    meta20 = data20[fn]['log'][tidx]['metadata']
                    meta21 = turn['metadata']
                    for dom, book_semi in meta21.items():
                        if 'semi' in book_semi:
                            semi21 = book_semi['semi']
                            if 'name' in semi21 and '|' in semi21['name']:
                                semi21['name'] = semi21['name'].split('|')[0]
                            if dom in meta20 and 'semi' in meta20[dom]:
                                semi20 = meta20[dom]['semi']
                                user20 = data20[fn]['log'][tidx-1]['text']
                                if 'name' in semi20 and 'name' in semi21 and semi21['name']!=semi20['name'] and semi20['name'] in user20:
                                    flag = False
                                    for d, entities in db.items():
                                        for ent in entities:
                                            if ent.get('name') and semi20['name'] == ent['name']:
                                                flag=True
                                                break
                                    if flag:
                                        name_map[semi21['name']] = semi20['name']
        del name_map['none']
        del name_map['taj tandoori']
        with open(self.save_path+'name_map.json', 'w', encoding='utf-8') as sf:
            json.dump(name_map,sf,indent=2)
        print('name mapping saved!')


    def preprocess_db(self, db_paths, otlg):
        # ensure the same value tokenization and slot normalization process as data
        db_values = {}
        value_to_slot_map= {}
        ambiguous_values = []

        dbs = {}
        for domain in otlg.db_domains:
            with open(db_paths[domain], 'r') as f:
                dbs[domain] = json.loads(f.read().lower())
                for idx, entry in enumerate(dbs[domain]):
                    new_entry = copy.deepcopy(entry)
                    for slot, value in entry.items():
                        # normalize entry
                        if type(value) is not str:
                            continue
                        del new_entry[slot]
                        value = value.replace('swimmingpool', 'swimming pool').replace('mutliple', 'multiple')
                        if slot in otlg.slot_normlize:
                            slot = otlg.slot_normlize[slot]
                        value_tknz_and_back = ' '.join(tknz(value)).strip()

                        new_entry[slot] = value_tknz_and_back
                        dbs[domain][idx] = new_entry

                        # extract informable slot values
                        v = value_tknz_and_back
                        if slot in otlg.informable_slots_dict[domain]:
                            if domain+'-'+slot not in db_values:
                                db_values[domain+'-'+slot] = [v]
                            elif v not in db_values[domain+'-'+slot]:
                                db_values[domain+'-'+slot].append(v)

                        # extract all values for delexicalization
                        if slot in otlg.informable_slots_dict[domain] + otlg.requestable_slots_dict[domain]:
                            if slot in ['parking', 'internet', 'phone', 'postcode', 'id', 'stars', 'price']:
                                continue
                            if v in value_to_slot_map and value_to_slot_map[v] != slot:
                                # print(value, ": ",value_to_slot_map[value], slot)
                                ambiguous_values.append(v)
                            value_to_slot_map[v] = slot

            print('[%s] DB processed! '%domain)

        ambiguous_values = list(set(ambiguous_values))
        for amb_v in ambiguous_values:   # departure or destination? arrive time or leave time?
            value_to_slot_map.pop(amb_v)
        value_to_slot_map['parkside'] = 'address'
        value_to_slot_map['parkside , cambridge'] = 'address'
        value_to_slot_map['hills road'] = 'address'
        value_to_slot_map['hills rd'] = 'address'
        value_to_slot_map['cambridge belfry'] = 'name'
        value_to_slot_map['parkside police station'] = 'name'
        del value_to_slot_map['hotel']
        for v in [ "toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen", "tesla"] + \
                    [ "black", "white", "red", "yellow", "blue", "grey" ]:
            value_to_slot_map[v] = 'car'
        ambiguous_values.remove('cambridge')

        single_token_values = {}
        multi_token_values = {}
        for val, slt in value_to_slot_map.items():
            if len(val.split())>1:
                multi_token_values[val] = slt
            else:
                single_token_values[val] = slt
                if slt == 'type':
                    single_token_values[val+'s'] = slt
        single_token_values['1000'] = 'choice'
        single_token_values['1029'] = 'choice'
        single_token_values['2828'] = 'choice'
        single_token_values['cb259aq'] = 'postcode'
        single_token_values['churches'] = 'type'
        multi_token_values['guest house'] = 'type'
        multi_token_values['arbury lodge guesthouse'] = 'name'
        multi_token_values['st . johns street'] = 'address'
        multi_token_values["st . john 's st."] = 'address'
        multi_token_values["st . johns"] = 'name'


        with open(self.save_path + 'db_processed.json', 'w', encoding='utf-8') as f:
            json.dump(dbs, f, indent=2)
        with open(self.save_path + 'db_values.json', 'w', encoding='utf-8') as f:
            json.dump(db_values, f, indent=2)
        with open(self.save_path + 'single_token_values.json', 'w', encoding='utf-8') as f:
            single_token_values = OrderedDict(sorted(single_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(single_token_values, f, indent=2)
        with open(self.save_path + 'multi_token_values.json', 'w', encoding='utf-8') as f:
            multi_token_values = OrderedDict(sorted(multi_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(multi_token_values, f, indent=2)
        with open(self.save_path + 'ambiguous_values.json', 'w', encoding='utf-8') as f:
            json.dump(ambiguous_values, f, indent=2)
        print('value dict saved!')


    def delexicalization(self, text, dialog_act=None, keep_list=[]):
        for value in self.reference_nos:
            text = text.replace(value, '[value_reference]')

        # delex by dialog act annotation
        if dialog_act is not None:
            text = ' ' + text + ' '
            delex_list = []
            for act, params in dialog_act.items():
                if 'request' in act or 'general' in act:
                    continue
                for s_v in params:
                    slot, value = s_v[0], s_v[1]
                    if slot != 'none' and value != 'none' and value not in keep_list:
                        delex_list.append([value, slot])
                        if number_to_text.get(value):
                            delex_list.append([number_to_text[value], slot])

            delex_list = sorted(delex_list, key=lambda x: len(x[0]), reverse=True)
            for s_v in delex_list:
                text = text.replace(' %s '%s_v[0], ' [value_%s] '%s_v[1], 1)
            text = text.strip()

        # delex by value dict: name, address, type, food etc
        for value, slot in self.delex_mt_valdict.items():
            if value not in keep_list:
                text = text.replace(value, '[value_%s]'%slot)
        for value, slot in self.delex_sg_valdict.items():
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value and value not in keep_list:
                    tokens[idx] = '[value_%s]'%slot
            text = ' '.join(tokens)

        # delex by rules: phone, stars, price, trainID, postcode
        text = re.sub(r'\d{5}\s?\d{5,7}', '[value_phone]', text)
        text = re.sub(r'\d[\s-]stars?', '[value_stars]', text)
        text = re.sub(r'\$\d+|\$?\d+.?(\d+)?\s(pounds?|gbps?)', '[value_price]', text)
        text = re.sub(r'tr[\d]{4}', '[value_id]', text)
        text = re.sub(r'there are (\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)', 'there are [value_choice]', text)
        text = re.sub(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', '[value_postcode]', text)

        # delex ambiguous values: arrive/leave time, departure/destination
        for ambg_ent in self.ambiguous_vals:
            start_idx = text.find(' '+ambg_ent)   # ely is a place, but appears in words like moderately
            if start_idx == -1 or ambg_ent in keep_list:
                continue
            front_words = text[:start_idx].split()
            ent_type = 'time' if ':' in ambg_ent else 'place'

            for fw in front_words[::-1]:
                if fw in ['arrive', 'arrives', 'arrived', 'arriving', 'arrival', 'destination', 'there', 'reach',  'to', 'by', 'before']:
                    slot = '[value_arrive]' if ent_type=='time' else '[value_destination]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)
                elif fw in ['leave', 'leaves', 'leaving', 'depart', 'departs', 'departing', 'departure',
                                'from', 'after', 'pulls']:
                    slot = '[value_leave]' if ent_type=='time' else '[value_departure]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)

        # clean
        text = text.replace('[value_car] [value_car]', '[value_car]')
        text = text.replace('[value_address] , [value_address] , [value_address]', '[value_address]')
        text = text.replace('[value_address] , [value_address]', '[value_address]')
        text = text.replace('[value_name] [value_name]', '[value_name]')
        text = text.replace('[value_name]([value_phone] )', '[value_name] ( [value_phone] )')
        return text


    def preprocess_main(self):
        """
        """
        data = {}
        count=0
        self.unique_da = {}
        for fn, raw_dial in tqdm(list(self.original_data.items())):
            count +=1
            # if count == 100:
            #     break
            compressed_goal = {}
            dial_domains, dial_reqs = [], []
            for dom, g in raw_dial['goal'].items():
                if dom != 'topic' and dom != 'message' and g:
                    compressed_goal[dom] = g
                    if g.get('reqt'):
                        for i, req_slot in enumerate(g['reqt']):
                            req_slot = self.otlg.slot_normlize.get(req_slot, req_slot)
                            dial_reqs.append(req_slot)
                    if dom in self.otlg.all_domains:
                        dial_domains.append(dom)

            dial_reqs = list(set(dial_reqs))

            dial = {'goal': compressed_goal, 'log': []}
            dial_state = {}
            prev_dial_state = {}
            prev_turn_domain = ['general']
            prev_user = ''
            single_turn = {}

            for turn_num, dial_turn in enumerate(raw_dial['log']):
                metadata = dial_turn['metadata']
                if not metadata:   # user
                    single_turn['user'] = ' '.join(tknz(clean_text(dial_turn['text'])))
                else:   #system
                    # get dialog state
                    keep_list = {}
                    name_from_db = ''
                    for domain in dial_domains:
                        if not dial_state.get(domain):
                            dial_state[domain] = {}
                        info_sv = metadata[domain]['semi']
                        for s,v in info_sv.items():
                            s = self.otlg.slot_normlize.get(s, s)
                            if len(v.split())>1:
                                v = self.name_map.get(v, v)
                                v = ' '.join(tknz(v))
                            if '|' in v:   # do not consider multiple names
                                v = v.replace('|',' | ').split('|')[0]
                            v = v.strip()
                            if v != '' and v != 'none' and v != 'not mentioned':
                                dial_state[domain][s] = v
                                keep_list[v] = 1
                                if domain+'-'+s not in self.otlg_values:
                                    self.otlg_values[domain+'-'+s] = [v]
                                elif v not in self.otlg_values[domain+'-'+s]:
                                    self.otlg_values[domain+'-'+s].append(v)
                                if s == 'name'  and domain in prev_dial_state:
                                    if s not in prev_dial_state[domain] and v not in prev_user + single_turn['user']:
                                        name_from_db = v
                        book_sv = metadata[domain]['book']
                        for s,v in book_sv.items():
                            if s == 'booked':
                                continue
                            s = self.otlg.slot_normlize.get(s, s)
                            if len(v.split())>1:
                                v = self.name_map.get(v, v)
                                v = ' '.join(tknz(v))
                            if '|' in v:   # do not consider multiple names
                                v = v.replace('|',' | ').split('|')[0]
                            v = v.strip()
                            if v != '' and v != 'none' and v != 'not mentioned':
                                dial_state[domain][s] = v
                                keep_list[v] = 1
                                if domain+'-'+s not in self.otlg_values:
                                    self.otlg_values[domain+'-'+s] = [v]
                                elif v not in self.otlg_values[domain+'-'+s]:
                                    self.otlg_values[domain+'-'+s].append(v)
                    # print(dial_state)

                    dial_state_flat = []
                    for domain, info_slots in dial_state.items():
                        if info_slots:
                            dial_state_flat.append('['+domain+']')
                            for slot, value in info_slots.items():
                                dial_state_flat.append(slot)
                                dial_state_flat.extend(value.split())


                    # get system dialog act and normalize
                    dialog_act = {}
                    inform, request = [], []
                    try:
                        original_dialog_act = self.dialog_acts[fn[:-5]][str(int((turn_num+1)/2))]
                    except:
                        # print(fn, turn_num)
                        original_dialog_act = None
                    if isinstance(original_dialog_act, dict):
                        for act, params in original_dialog_act.items():
                            dialog_act[act] = []
                            for s_v in params:
                                slot = self.otlg.slot_normlize.get(s_v[0], s_v[0])
                                value = ' '.join(tknz(s_v[1])).strip()
                                if slot in ['leave', 'arrive', 'time']:
                                    value = clean_time(value)
                                if 'minute' in value:
                                    slot = 'duration'
                                dialog_act[act].append([slot, value])
                                if 'request' in act and slot not in request:
                                    request.append(slot)
                                elif 'offerbook' in act and 'book' not in request:
                                    request.append('book')
                                elif slot != 'none' and slot not in inform and slot not in ['parking', 'internet']:
                                    inform.append('[value_%s]'%slot)

                    resp = ' '.join(tknz(clean_text(dial_turn['text'])))
                    resp_delex_all = self.delexicalization(resp, dialog_act)
                    single_turn['resp'] = resp_delex_all
                    single_turn['resp_ori'] = resp #self.delexicalization(resp, dialog_act, keep_list)
                    single_turn['name_from_db'] = name_from_db


                    # ordered system act
                    # request_ordered = {}
                    # for slot in request:
                    #     request_ordered[slot] = resp_delex_all.find(slot)
                    #     if resp_delex_all.find(slot) == -1:
                    #         print(fn, turn_num)
                    #         print(resp_delex_all)
                    #         print(slot)
                    # request_ordered = sorted(request_ordered.items(), key=lambda x:x[1])
                    request_ordered = request
                    inform_ordered = {}
                    for slot in inform:
                        inform_ordered[slot] = resp_delex_all.find(slot)
                        # if resp_delex_all.find(slot) == -1:
                        #     print(fn, turn_num)
                        #     print(resp_delex_all)
                        #     print(slot)
                    inform_ordered = sorted(inform_ordered.items(), key=lambda x:x[1])
                    inform_ordered = [s[0] for s in inform_ordered]

                    # get turn domain
                    turn_dom_bs = []
                    for dom, info_slots in dial_state.items():
                        if info_slots:
                            if dom not in prev_dial_state or prev_dial_state[dom] != dial_state[dom]:
                                turn_dom_bs.append(dom)

                    turn_dom_da = set()
                    for act in dialog_act:
                        d, a = act.split('-')
                        turn_dom_da.add(d)
                    turn_dom_da = list(turn_dom_da)
                    if len(turn_dom_da) != 1 and 'general' in turn_dom_da:
                        turn_dom_da.remove('general')
                    if len(turn_dom_da) != 1 and 'booking' in turn_dom_da:
                        turn_dom_da.remove('booking')

                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]

                    # get db pointers
                    matnums = self.db.get_match_num(dial_state)
                    match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
                    match = matnums[match_dom]
                    dbvec = self.db.addDBPointer(match_dom, match)
                    bkvec = self.db.addBookingPointer(dialog_act)

                    single_turn['pointer'] = ','.join([str(d) for d in dbvec + bkvec])
                    single_turn['match'] = str(match)
                    single_turn['constraint'] = json.dumps(dial_state),
                    single_turn['sys_inform'] = ' '.join(inform_ordered)
                    single_turn['sys_request'] = ' '.join(request_ordered)
                    single_turn['turn_num'] = len(dial['log'])
                    single_turn['turn_domain'] = ' '.join(turn_domain)


                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_dial_state = copy.deepcopy(dial_state)
                    if 'user' in single_turn:
                        prev_user = copy.deepcopy(single_turn['user'])
                        dial['log'].append(single_turn)
                        for t in single_turn['user'].split() + single_turn['resp'].split():
                            self.vocab.add_word(t)

                    single_turn = {}

            data[fn] = dial
            # pprint(dial)
            # if count == 20:
            #     break
        self.vocab.construct()
        self.vocab.save_vocab(self.save_path + 'vocab')
        with open(self.save_path + 'data_processed.json', 'w') as f:
            json.dump(data, f, indent=2)
        with open(self.save_path + 'ontology_values.json', 'w') as f:
            json.dump(self.otlg_values, f, indent=2)

        return data




if __name__ == '__main__':
    dp = DataPreprocessor()
