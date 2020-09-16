import logging, time, os, json, random, argparse
import numpy as np
import torch

import utils
from config import global_config as cfg
from reader import CamRest676Reader, MultiwozReader, KvretReader
from vae_model import SemiCVAE
from smc_model import SemiBootstrapSMC
# from cjsa_model import SemiNASMC
# from rl_model import SemiRL
from metric import CamRestEvaluator, MultiwozEvaluator, KvretEvaluator

model_mapping = {
    'cvae': SemiCVAE,
    'bssmc': SemiBootstrapSMC,
    # 'nasmc': SemiNASMC,
    # 'rl': SemiRL,
}

class Model:
    def __init__(self, dataset, method):
        self.dataset = dataset
        if dataset == 'camrest':
            self.reader = CamRest676Reader()
            self.evaluator = CamRestEvaluator(self.reader)
        elif dataset == 'multiwoz':
            self.reader = MultiwozReader()
            self.evaluator = MultiwozEvaluator(self.reader)
        elif dataset == 'kvret':
            self.reader = KvretReader()
            self.evaluator = KvretEvaluator(self.reader)
        else:
            raise NotImplementedError('Other datasets to be implemented !')

        self.method = method
        self.model_fun = model_mapping.get(self.method, None)
        if self.model_fun is None:
            raise ValueError('Unimplemented method')
        self.model = self.model_fun(cfg=cfg, reader=self.reader)
        if cfg.cuda: self.model = self.model.cuda()

        self.unsup_prop = 0 if cfg.skip_unsup else 100 - cfg.spv_proportion
        self.final_epoch = 0

    def _convert_batch(self, py_batch):
        u_input_np = utils.padSeqs(py_batch['user'], cfg.u_max_len, truncated=True)
        m_input_np = utils.padSeqs(py_batch['resp'], cfg.m_max_len, truncated=True)
        db_vec = torch.from_numpy(np.array(py_batch['db_vec'])).float()
        filling_vec = torch.from_numpy(np.array(py_batch['filling_vec'])).float()
        u_input = torch.from_numpy(u_input_np).long()
        m_input = torch.from_numpy(m_input_np).long()
        if cfg.cuda:
            u_input, m_input, db_vec = u_input.cuda(), m_input.cuda(), db_vec.cuda()
            filling_vec = filling_vec.cuda()
        z_input = {}
        for sn in self.reader.otlg.informable_slots:
            # eos_idx = self.reader.vocab.encode(sn)
            z_input_py = [state[sn] for state in py_batch['bspn']]
            z_input_np = utils.padSeqs(z_input_py, maxlen=cfg.z_length, fixed_length=True)#,  value=eos_idx)
            # z_input_np = utils.padSeqs(z_input_py, maxlen=cfg.z_length)#,  value=eos_idx)
            z_input[sn] = torch.from_numpy(z_input_np).long()
            if cfg.cuda:
                z_input[sn] = z_input[sn].cuda()
        if cfg.model_act:
            a_input = {}
            for sn in self.reader.act_order:
                a_input_py = [act[sn] for act in py_batch['aspn']]
                a_input_np = utils.padSeqs(a_input_py, maxlen=cfg.a_length, fixed_length=True)#,  value=eos_idx)
                # a_input_np = utils.padSeqs(a_input_py, maxlen=cfg.a_length)#,  value=eos_idx)
                a_input[sn] = torch.from_numpy(a_input_np).long()
                if cfg.cuda:
                    a_input[sn] = a_input[sn].cuda()
        else:
            a_input = {}
        supervised = py_batch['supervised'][0]
        return u_input, z_input, a_input, m_input, db_vec, filling_vec, supervised


    def train(self, pretrain=False, skip_sup=False):
        self.final_epoch = 0
        prev_min_loss, early_stop_count = 1e9, cfg.early_stop_count
        weight_decay_count = cfg.weight_decay_count
        st = time.time()
        lr = cfg.lr
        optim = torch.optim.Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.model.parameters()))
        # optim = torch.optim.SGD(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.model.parameters()))
        train_loss_col, valid_loss_col, valid_score_col, event_col = [], [], [], []
        max_epoch = cfg.sup_pretrain if pretrain else cfg.max_epoch
        mode = 'train' if cfg.mode != 'rl_tune' else 'rl_tune'
        for epoch in range(max_epoch):
            logging.info('************ epoch: %d ************' % (epoch))
            self.model.self_adjust(epoch)
            epoch_sup_loss, epoch_unsup_loss = {'loss':0 }, {'loss':0}
            sup_cnt, unsup_cnt = 0, 0
            random.shuffle(self.reader.batches['train'])
            for iter_num,dial_batch in enumerate(self.reader.batches['train']):
                turn_states = {}
                for turn_num, turn_batch in enumerate(dial_batch):
                    # print('epoch:', epoch, 'dial:', iter_num,'turn:', turn_num)
                    optim.zero_grad()
                    u_input, z_input, a_input, m_input, db_vec, filling_vec, supervised = \
                            self._convert_batch(turn_batch)
                    if cfg.multi_domain:
                        turn_states['dom'] = turn_batch['dom']
                    if supervised:
                        if skip_sup:
                            continue
                        losses, turn_states = self.model(u_input=u_input, z_input=z_input, m_input=m_input,
                                                                  db_vec=db_vec, filling_vec=filling_vec,
                                                                  z_supervised=True, turn_states=turn_states, mode=mode,
                                                                  a_input=a_input)
                        total_loss = losses['loss']
                        if cfg.prev_z_continuous:
                            total_loss.backward(retain_graph=turn_num!=len(dial_batch)-1)
                        else:
                            total_loss.backward(retain_graph=False)
                        # grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0)
                        optim.step()
                        # torch.cuda.empty_cache()
                        for k, v in losses.items():
                            if k == 'reward': continue
                            epoch_sup_loss[k]=v.item() if k not in epoch_sup_loss else epoch_sup_loss[k]+v.item()
                        sup_cnt += 1

                        loss_log = ''
                        for k in ['pz', 'qz', 'pa', 'qa', 'm', 'kl', 'kl_a']:
                            if k+'_loss' in losses:
                                loss_log += '|%s: %.3f'%(k, losses[k+'_loss'].item())
                        if 'reward' in losses:
                            loss_log += '|reward: %.5f'%(losses['reward'])
                        if cfg.dataset != 'multiwoz' or iter_num %5 == 0:
                            logging.info( '%d super loss: %.3f%s'%(iter_num, total_loss.item(), loss_log))

                    else:
                        if cfg.skip_unsup or pretrain:
                            #logging.info('skipping unsupervised batch')
                            continue
                        losses, turn_states = self.model(u_input=u_input, z_input=z_input, m_input=m_input,
                                                              z_supervised=False, turn_states=turn_states, mode=mode,
                                                              a_input=a_input, db_vec=db_vec, filling_vec=filling_vec,
                                                              no_label_train=cfg.no_label_train)
                        total_loss = losses['loss']
                        if cfg.prev_z_continuous:
                            total_loss.backward(retain_graph=turn_num!=len(dial_batch)-1)
                        else:
                            total_loss.backward(retain_graph=False)
                        # grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(),4.0)
                        optim.step()
                        for k, v in losses.items():
                            if k == 'reward': continue
                            epoch_unsup_loss[k] = v.item() if k not in epoch_unsup_loss else epoch_unsup_loss[k]+v.item()
                        unsup_cnt += 1
                        loss_log = ''
                        for k in ['pz', 'qz', 'pa', 'qa', 'm']:
                            if k+'_loss' in losses:
                                loss_log += '|%s: %.3f'%(k, losses[k+'_loss'].item())
                        for k in ['kl', 'kl_a']:
                            if k+'_loss' in losses:
                                loss_log += '|%s: %.5f'%(k, losses[k+'_loss'].item())
                        if 'reward' in losses:
                            loss_log += '|reward: %.5f'%(losses['reward'])
                        if cfg.dataset != 'multiwoz' or iter_num %5 == 0:
                            logging.info( '%d unsup loss: %.3f%s'%(iter_num, total_loss.item(), loss_log))

                        # if epoch>5:
                        # print(turn_states['norm_W'])
            train_loss = {}
            for k in epoch_sup_loss:
                train_loss['sup_' + k] = epoch_sup_loss[k] / (sup_cnt + 1e-8)
            if 'm_loss' in epoch_sup_loss and 'pz_loss' in epoch_sup_loss:
                train_loss['sup_p_joint_nll'] = train_loss['sup_m_loss'] + train_loss['sup_pz_loss']
            for k in epoch_unsup_loss:
                train_loss['unsup_' + k] = epoch_unsup_loss[k] / (unsup_cnt + 1e-8)
            if 'm_loss' in epoch_unsup_loss and 'pz_loss' in epoch_unsup_loss:
                train_loss['unsup_p_joint_nll'] = train_loss['unsup_m_loss'] + train_loss['unsup_pz_loss']

            logging.info('epoch: %d  train loss sup: %.3f  unsup: %.3f  total time: %.1fmin' %(
                epoch+1, train_loss['sup_loss'], train_loss['unsup_loss'], (time.time()-st)/60))

            # do validation
            valid_loss, valid_score = self.validate()
            train_loss_col.append(train_loss)
            valid_loss_col.append(valid_loss)
            valid_score_col.append(valid_score)

            logging.info('*************** RUNNING TEST ***************')
            _, _ = self.validate(data='test')

            if 'loss' in cfg.valid_type:
                valid_loss_check = valid_loss[cfg.valid_type]
                if cfg.model_act and cfg.valid_type == 'loss':
                    valid_loss_check -= valid_loss['pa_loss']
            else:
                valid_loss_check = - valid_score[cfg.valid_type]   #scores are the higher the better

            if valid_loss_check < prev_min_loss:
                early_stop_count = cfg.early_stop_count
                weight_decay_count = cfg.weight_decay_count
                if epoch>=cfg.min_epoch:
                    prev_min_loss = valid_loss_check
                    best_valid_loss = valid_loss
                    self.save_model(epoch)
                    if epoch>10 and not cfg.save_log:   #only for debug
                        logging.info('*************** RUNNING TEST ***************')
                        _, _ = self.validate(data='test')
                event_col.append('')
            else:
                early_stop_count -= 1
                weight_decay_count -= 1
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))
                if not early_stop_count:
                    event_col.append('early stop')
                    self.final_epoch = epoch+1
                    break
                if not weight_decay_count:
                    lr *= cfg.lr_decay
                    for group in optim.param_groups:
                        group['lr'] = lr
                    weight_decay_count = cfg.weight_decay_count
                    logging.info('learning rate decay, learning rate: %f' % (lr))
                    event_col.append('lr decay: %.6f'%lr)
                else:
                    event_col.append('')

        if cfg.save_log and max_epoch != 0:   # Do evaluation and save result report
            self.load_model()
            self.final_epoch = epoch+1
            logging.info('Total training epoch number: %d' % (self.final_epoch))
            if pretrain:
                self.save_model(epoch, path=os.path.join(cfg.exp_path, 'pretrained_model.pkl'))
                pretrain_result_save = os.path.join(cfg.eval_load_path, 'sup_pretrained_results.csv')
                metric_results = self.eval(result_save_path=pretrain_result_save)
                report_path = cfg.global_record_path[:-4] + '_pretrain.csv'
                self.reader.save_loss(train_loss_col, valid_loss_col, event_col, file_name='loss_pretrain.csv')
            else:
                file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'eval_log.json'))
                logging.getLogger('').addHandler(file_handler)
                logging.info(str(cfg))
                metric_results = self.eval(result_save_path=cfg.result_path)
                report_path = cfg.global_record_path
                self.reader.save_loss(train_loss_col, valid_loss_col, event_col, file_name='loss.csv')
            metric_results['final_train_loss'] = train_loss
            metric_results['best_valid_loss'] = best_valid_loss

            self.reader.save_result_report(metric_results, report_path)


    def validate(self, data='dev'):
        self.model.eval()
        result_collection = {}
        with torch.no_grad():
            loss_col, count = {}, 0
            logging.info('begin validation')
            for dial_batch in self.reader.batches[data]:
                turn_states, turn_states_loss = {}, {}
                for turn_num, turn_batch in enumerate(dial_batch):
                    u_input, z_input, a_input, m_input, db_vec, filling_vec, supervised = \
                            self._convert_batch(turn_batch)
                    if cfg.multi_domain:
                        turn_states_loss['dom'] = turn_batch['dom']
                        turn_states['dom'] = turn_batch['dom']
                    # print('turn_num:', turn_num)
                    losses, turn_states_loss = self.model(u_input=u_input, z_input=z_input, m_input=m_input,
                                                                  z_supervised=True, turn_states=turn_states_loss, mode='loss_eval',
                                                                  a_input=a_input, db_vec=db_vec, filling_vec=filling_vec)

                    for k, v in losses.items():
                        if k == 'reward': continue
                        loss_col[k] = v.item() if k not in loss_col else loss_col[k] + v.item()
                    count += 1
                    index, db, turn_states = self.model(mode='test', u_input=u_input, z_input=z_input,
                                                          m_input=m_input, z_supervised=None, turn_states=turn_states,
                                                          a_input=a_input, db_vec=db_vec, filling_vec=filling_vec,)
                    turn_batch['resp_gen'], turn_batch['bspn_gen'], turn_batch['db_gen'] = index['m_idx'], index['z_idx'], db
                    if cfg.model_act:
                        turn_batch['aspn_gen'] = index['a_idx']
                #     print(self.reader.vocab.sentence_decode(z_input['food'][0]),self.reader.vocab.sentence_decode(z_input['pricerange'][0]),self.reader.vocab.sentence_decode(z_input['area'][0]))
                #     print(self.reader.vocab.sentence_decode(z_idx[0]))
                #     print('')
                #     print(self.reader.vocab.sentence_decode(z_input['food'][1]),self.reader.vocab.sentence_decode(z_input['pricerange'][1]),self.reader.vocab.sentence_decode(z_input['area'][1]))
                #     print(self.reader.vocab.sentence_decode(z_idx[1]))
                # print('')
                result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

            for k in loss_col:
                loss_col[k] /= (count + 1e-8)
            loss_log = ''
            for k in ['pz', 'qz', 'pa', 'qa', 'm', 'kl', 'kl_a']:
                if k+'_loss' in losses: loss_log += '|%s: %.3f'%(k, loss_col[k+'_loss'])
            logging.info( 'valid loss: %.3f%s'%(loss_col['loss'], loss_log))

            logging.info('validation metric scores:')
            results, field = self.reader.wrap_result(result_collection)
            metrics = self.evaluator.run_metrics(results)
            combined = metrics['bleu'] + metrics['match']/3 + metrics['joint_goal']
            score = {'bleu': metrics['bleu'], 'match': metrics['match'], 'joint_goal': metrics['joint_goal'],
                          'combined': combined}
            # metrics['epoch_num'] = self.final_epoch
            # metrics['test_loss'] = loss_col
            # metric_field = list(metrics.keys())
            # self.reader.save_result('w', [metrics], metric_field, write_title='EVALUATION RESULTS:',
            #                                     result_save_path=cfg.result_path)
            # self.reader.save_result('a', results, field, write_title='DECODED RESULTS:',
            #                                     result_save_path=cfg.result_path)
        # self.eval()
        self.model.train()
        return loss_col, score


    def eval(self, data='test', result_save_path=None):
        result_collection = {}
        with torch.no_grad():
            self.model.eval()
            logging.info('begin testing')
            loss_col, count = {}, 0
            for batch_num, dial_batch in enumerate(self.reader.batches[data]):
                turn_states, turn_states_loss = {}, {}
                for turn_batch in dial_batch:
                    u_input, z_input, a_input, m_input, db_vec, filling_vec, supervised = \
                            self._convert_batch(turn_batch)
                    if cfg.multi_domain:
                        turn_states_loss['dom'] = turn_batch['dom']
                        turn_states['dom'] = turn_batch['dom']
                    losses, turn_states_loss = self.model(u_input=u_input, z_input=z_input, m_input=m_input,
                                                                  z_supervised=True, turn_states=turn_states_loss, mode='loss_eval',
                                                                  a_input=a_input, db_vec=db_vec, filling_vec=filling_vec)
                    for k, v in losses.items():
                        loss_col[k] = v.item() if k not in loss_col else loss_col[k] + v.item()
                    count += 1
                    index, db, turn_states = self.model(mode='test', u_input=u_input, z_input=z_input,
                                                          m_input=m_input, z_supervised=None, turn_states=turn_states,
                                                          a_input=a_input, db_vec=db_vec, filling_vec=filling_vec)
                    turn_batch['resp_gen'], turn_batch['bspn_gen'], turn_batch['db_gen'] = index['m_idx'], index['z_idx'], db
                    if cfg.model_act:
                        turn_batch['aspn_gen'] = index['a_idx']
                # print('{}\r'.format(batch_num))
                result_collection.update(self.reader.inverse_transpose_batch(dial_batch))
        for k in loss_col:
            loss_col[k] /= (count + 1e-8)
        results, field = self.reader.wrap_result(result_collection)
        metric_results = self.evaluator.run_metrics(results)
        metric_results['epoch_num'] = self.final_epoch
        metric_results['test_loss'] = loss_col
        metric_field = list(metric_results.keys())
        self.reader.save_result('w', [metric_results], metric_field, write_title='EVALUATION RESULTS:',
                                            result_save_path=result_save_path)
        self.reader.save_result('a', results, field, write_title='DECODED RESULTS:',
                                            result_save_path=result_save_path)
        self.model.train()
        return metric_results



    def save_model(self, epoch, path=None):
        if not cfg.save_log:
            return
        if not path:
            path = cfg.model_path
        all_state =self.model.state_dict()
        torch.save(all_state, path)
        logging.info('Model saved')

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(all_state)
        logging.info('Model loaded')

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.model.u_encoder.embedding.weight.data.cpu().numpy()
        if cfg.embed_size == 50:
            glove_path = './data/glove/glove_multiwoz.6B.50d.txt'
        elif cfg.embed_size == 300:
            glove_path = './data/glove/glove_multiwoz.840B.300d.txt'
        else:
            return
        embedding_arr = self.reader.get_glove_matrix(glove_path, initial_arr)
        embedding_arr = torch.from_numpy(embedding_arr)
        self.model.embedding.weight.data.copy_(embedding_arr)
        if freeze:
            self.freeze_module(self.model.embedding)

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))
        logging.info('total trainable params: %d' % param_cnt)
        return param_cnt

def parse_arg_cfg(args):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-method')
    parser.add_argument('-dataset')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    if not os.path.exists('experiments/'+args.dataset):
        os.mkdir('experiments/'+args.dataset)
    if not os.path.exists('log/'+args.dataset):
        os.mkdir('log/'+args.dataset)

    cfg.mode, cfg.dataset, cfg.method = args.mode, args.dataset, args.method
    cfg.init_handler(cfg.dataset)
    if args.mode in ['test', 'adjust', 'rl_tune']:
        parse_arg_cfg(args)
        cfg_load = json.loads(open(os.path.join(cfg.eval_load_path, 'config.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_load_path', 'log_time', 'model_path',
                        'result_path', 'model_parameters', 'beam_search', 'skip_unsup', 'sup_pretrain',
                        'lr', 'valid_type', 'seed', 'max_epoch']:
                continue
            setattr(cfg, k, v)
            cfg.model_path = os.path.join(cfg.eval_load_path, 'model.pkl')
            result_file = 'result_beam.csv' if cfg.beam_search else 'result_greedy.csv'
            # cfg.model_path = os.path.join(cfg.eval_load_path, 'model_rl.pkl')
            # result_file = 'result_beam_rl.csv' if cfg.beam_search else 'result_greedy_rl.csv'
            cfg.result_path = os.path.join(cfg.eval_load_path, result_file)
            cfg.vocab_path = os.path.join(cfg.eval_load_path, 'vocab')
    else:
        parse_arg_cfg(args)
        if cfg.exp_path in ['' , 'to be generated']:
            unsup_prop = 0 if cfg.skip_unsup else 100 - cfg.spv_proportion
            cfg.exp_path = 'experiments/{}/{}_{}_sd{}_sup{}un{}_zl{}_lr{}_dp{}/'.format(
                cfg.dataset, cfg.method, cfg.exp_no, cfg.seed, cfg.spv_proportion, unsup_prop,
                cfg.z_length, cfg.lr, cfg.dropout_rate)
            if cfg.save_log:
                os.mkdir(cfg.exp_path)
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.eval_load_path = cfg.exp_path

    cfg.init_logging_handler(args.mode)
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.info('Device: {}'.format(torch.cuda.current_device()))

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    m = Model(args.dataset, args.method)
    logging.info(str(cfg))
    m.count_params()

    if args.mode == 'train':
        if cfg.save_log:
            m.reader.vocab.save_vocab(os.path.join(cfg.exp_path, 'vocab'))
            with open(os.path.join(cfg.exp_path, 'config.json'), 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        if cfg.glove_init:
            m.load_glove_embedding(freeze=cfg.freeze_emb)
        if unsup_prop>0:
            m.train(pretrain=True)
            logging.info('Start semi-supervised training')
            if cfg.changedp and not cfg.use_resp_dpout:
                cfg.use_resp_dpout = True
            m.train()
        else:
            m.train()
    elif args.mode == 'adjust':
        m.load_model(cfg.model_path)
        m.train()
    elif args.mode == 'rl_tune':
        m.load_model(cfg.model_path)
        cfg.model_path = cfg.model_path.replace('.pkl', '_rl.pkl')
        cfg.result_path = os.path.join(cfg.eval_load_path, 'result_rl.csv')
        for mod in [m.model.embedding, m.model.u_encoder, m.model.z_encoder, m.model.pz_decoder, m.model.u_encoder_q]:
            m.freeze_module(mod)

        m.train()
    elif args.mode == 'test':
        m.load_model(cfg.model_path)
        metric_results = m.eval(result_save_path=cfg.result_path)
        m.reader.save_result_report(metric_results, cfg.global_record_path)


if __name__ == '__main__':
    main()
