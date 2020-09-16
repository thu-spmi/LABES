import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from config import global_config as cfg
from modules import get_one_hot_input, cuda_
from base_model import BaseModel
from utils import toss_

torch.set_printoptions(sci_mode=False)

class SemiBootstrapSMC(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(has_qnet=False, **kwargs)
        self.weight_normalize = nn.Softmax(dim=0)
        self.particle_num = cfg.particle_num

    def forward(self, u_input, m_input, z_input, a_input, turn_states, z_supervised, mode,
                        db_vec=None, filling_vec=None, no_label_train=False):
        if mode == 'train' or mode == 'loss_eval':
            debug = {'true_z': z_input, 'true_db': db_vec, 'true_a': a_input}
            if not z_supervised:
                z_input = None
                if not no_label_train:
                    u_input = torch.cat([u_input]*self.particle_num, dim=0)
                    m_input = torch.cat([m_input]*self.particle_num, dim=0)
            probs, index, turn_states = \
                self.forward_turn(u_input, m_input=m_input, z_input=z_input, is_train=True,
                                  turn_states=turn_states, db_vec=db_vec, debug=debug, mode=mode,
                                  a_input=a_input, filling_vec=filling_vec, no_label_train=no_label_train)
            if z_supervised:
                z_input = torch.cat(list(z_input.values()), dim=1)
                a_input = torch.cat(list(a_input.values()), dim=1) if cfg.model_act else None
                index.update({'z_input': z_input, 'a_input': a_input, 'm_input': m_input})
                loss, pz_loss, pa_loss, m_loss= self.supervised_loss(probs, index)
                losses = {'loss': loss, 'pz_loss': pz_loss, 'm_loss': m_loss}
                if cfg.model_act:
                    losses.update({'pa_loss': pa_loss})
                return losses, turn_states
            else:
                index.update({'m_input': m_input})
                if not no_label_train:
                    loss, pz_loss, pa_loss, m_loss= self.unsupervised_loss(probs, index, turn_states['norm_W'])
                    losses = {'loss': loss, 'pz_loss': pz_loss, 'm_loss': m_loss}
                    if cfg.model_act:
                        losses.update({'pa_loss': pa_loss})
                else:
                    loss, pz_loss, pa_loss, m_loss= self.supervised_loss(probs, index, no_label_train)
                    losses = {'loss': loss, 'm_loss': m_loss}

                return losses, turn_states

        elif mode == 'test':
            index, db, turn_states = self.forward_turn(u_input, is_train=False, a_input=a_input,
                                                                      turn_states=turn_states, db_vec=db_vec)
            return index, db, turn_states


    def forward_turn(self, u_input, turn_states, is_train, m_input=None, z_input=None,
                                a_input=None, db_vec=None, filling_vec=None, debug=None, mode=None,
                                no_label_train=False):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_len:
        :param turn_states:
        :param is_train:
        :param u_input: [B,T]
        :param m_input: [B,T]
        :param z_input: [B,T]
        :param: norm_W: [B, K]

        pv_pz_pr: K * [B,T,V]
        pv_pz_h: K * [B,T,H]
        :return:
        """
        batch_size = u_input.size(0)
        u_hiddens, u_last_hidden = self.u_encoder(u_input)
        u_input_1hot = get_one_hot_input(u_input, self.vocab_size)


        if is_train and z_input is None:   # unsupervised training
            if not no_label_train:
                ori_batch_size = int(u_input.size(0) / self.particle_num)
                norm_W = turn_states.get('norm_W', None)
                if norm_W is not None and cfg.resampling:  # Resampling
                    dis = Categorical(torch.cat([norm_W]*self.particle_num, dim=0)) # [B*K, K]
                    Ak = dis.sample()   #[B*K]
                    # print('Ak:', Ak.contiguous().view(self.particle_num,-1))
                    bias = np.tile(np.arange(0, ori_batch_size), self.particle_num)
                    idx = bias + Ak.cpu().numpy() * ori_batch_size
                    turn_states['pv_pz_h'] =  turn_states['pv_pz_h'][idx]    # [T, B*K, V]
                    turn_states['pv_pz_pr'] = turn_states['pv_pz_pr'][idx]  # [T, B*K, H]
                    turn_states['pv_pz_id'] = turn_states['pv_pz_id'][idx]

            sample_type = 'topk'
        elif is_train and z_input is not None and mode != 'loss_eval':   # supervised training
            sample_type = 'supervised'
        else:   #testing
            sample_type = 'top1'

        # P(z|pv_z, u)
        pz_prob, pz_samples, z_hiddens, turn_states, log_pz = \
            self.decode_z(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, z_input,
                                    turn_states, sample_type=sample_type, decoder_type='pz')

        # DB indicator and slot filling indicator
        if cfg.dataset == 'camrest':
            db_vec_np, match = self.db_op.get_db_degree(pz_samples, self.vocab)
            db_vec = cuda_(torch.from_numpy(db_vec_np).float())
        elif cfg.dataset == 'multiwoz':
            db_vec_np, match = self.db_op.get_db_degree(pz_samples, turn_states['dom'], self.vocab)
            db_vec_new = cuda_(torch.from_numpy(db_vec_np).float())
            db_vec[:, :4] = db_vec_new
        else:
            match = [0] * batch_size
        filling_vec = self.reader.cons_tensors_to_indicator(pz_samples)
        filling_vec = cuda_(torch.from_numpy(filling_vec).float())

        # P(a|u, db, slot_filling_indicator)
        if self.model_act:
            pa_prob, pa_samples, a_hiddens, log_pa = \
                self.decode_a(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, a_input,
                            db_vec, filling_vec, sample_type=sample_type, decoder_type='pa')
        else:
            pa_prob, pa_samples, a_hiddens = None, None, None

         # P(m|u, z, a ,db)
        if is_train or not self.beam_search:
            pm_prob, m_idx, log_pm = \
                    self.decode_m(batch_size, u_last_hidden, u_input, u_hiddens, u_input_1hot,
                                pz_samples, pz_prob, z_hiddens, pa_samples, pa_prob, a_hiddens,
                                db_vec, m_input, is_train=is_train)
        else:
            m_idx = self.beam_search_decode(u_input, u_input_1hot, u_hiddens, pz_samples,
                                                                    pz_prob, z_hiddens, db_vec, u_last_hidden[:-1],
                                                                    pa_samples, pa_prob, a_hiddens)

        # compute normalized weights W for unsupervised training
        if is_train and z_input is None and not no_label_train:
            log_w = log_pm
            log_w = log_w.view(self.particle_num, -1)
            norm_W = self.weight_normalize(log_w).transpose(1,0)    #[B,K]
            turn_states['norm_W'] = norm_W

        # output
        if is_train:
            probs = {'pz_prob': pz_prob, 'pm_prob': pm_prob, 'pa_prob': pa_prob}
            index = {'z_input': pz_samples, 'a_input': pa_samples}
            return probs, index, turn_states
        else:
            z_idx = self.max_sampling(pz_prob)
            a_idx = self.max_sampling(pa_prob) if self.model_act else None
            index = {'m_idx': m_idx, 'z_idx': z_idx, 'a_idx': a_idx}
            return index, match, turn_states

        # z_gt = debug['true_z']
        # print('u true:', self.vocab.sentence_decode(u_input[0], eos='<eos_u>'))
        # print('m true:', self.vocab.sentence_decode(m_input[0], eos='<eos_r>'))
        # # print('z true:', self.vocab.sentence_decode(z_gt['food'][0]),self.vocab.sentence_decode(z_gt['pricerange'][0]),self.vocab.sentence_decode(z_gt['area'][0]))
        # print('z samples:')
        # print(self.vocab.sentence_decode(pz_samples[0]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size*2]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size*3]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size*4]))
        # if pv_pz_pr is not None:
        #     print('Ak:', Ak[0].item(), Ak[batch_size].item(), Ak[batch_size*2].item(), Ak[batch_size*3].item(), Ak[batch_size*4].item())
        # print('W:', norm_W[0])
        # print(self.vocab.sentence_decode(pz_samples[1]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size+1]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size*2+1]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size*3+1]))
        # print(self.vocab.sentence_decode(pz_samples[batch_size*4+1]))
        # print('W:', norm_W[1])
        return probs, index, turn_states


    def supervised_loss(self, probs, index, no_label_train=False):
        pz_prob, pm_prob = torch.log(probs['pz_prob']), torch.log(probs['pm_prob'])
        z_input, m_input = index['z_input'], index['m_input']
        pz_loss = self.nll_loss(pz_prob.view(-1, pz_prob.size(2)), z_input.view(-1))
        m_loss = self.nll_loss(pm_prob.view(-1, pm_prob.size(2)), m_input.view(-1))
        if self.model_act:
            pa_prob =  torch.log(probs['pa_prob'])
            a_input = index['a_input']
            pa_loss = self.nll_loss(pa_prob.view(-1, pa_prob.size(2)), a_input.view(-1))
            loss = cfg.pz_loss_weight * pz_loss + m_loss + pa_loss
        else:
            pa_loss = torch.zeros(1)
            loss = cfg.pz_loss_weight * pz_loss + m_loss
        if no_label_train:
            loss = m_loss
        return loss, pz_loss, pa_loss, m_loss

    def unsupervised_loss(self, probs, index, norm_W):
        # pz_prob: [B*K, T, V]
        pz_prob, pm_prob = torch.log(probs['pz_prob']), torch.log(probs['pm_prob'])
        z_input, m_input = index['z_input'], index['m_input']
        if self.model_act:
            pa_prob, a_input =  torch.log(probs['pa_prob']), index['a_input']
        if cfg.weighted_grad:
            cliped_norm_W = torch.clamp(norm_W, min=1e-10, max=1)
            W = cliped_norm_W.transpose(1,0).contiguous().view(-1).detach()
            pz_prob = pz_prob.transpose(2,0) * W * cfg.particle_num
            pm_prob = pm_prob.transpose(2,0) * W * cfg.particle_num
            pz_prob, pm_prob = pz_prob.transpose(2,0).contiguous(), pm_prob.transpose(2,0).contiguous()
            if self.model_act:
                pa_prob = pa_prob.transpose(2,0) * W * cfg.particle_num
                pa_prob = pa_prob.transpose(2,0).contiguous()

        pz_loss = self.nll_loss(pz_prob.view(-1, pz_prob.size(2)), z_input.view(-1))
        m_loss = self.nll_loss(pm_prob.view(-1, pm_prob.size(2)), m_input.view(-1))
        if self.model_act:
            pa_loss = self.nll_loss(pa_prob.view(-1, pa_prob.size(2)), a_input.view(-1))
            loss = cfg.pz_loss_weight * pz_loss + m_loss + pa_loss
        else:
            pa_loss = torch.zeros(1)
            loss = cfg.pz_loss_weight * pz_loss + m_loss
        return loss, pz_loss, pa_loss, m_loss
