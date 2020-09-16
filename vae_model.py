import numpy as np
import torch
from torch import nn

from config import global_config as cfg
from modules import get_one_hot_input, cuda_
from base_model import BaseModel
from metric import BLEUScorer

class MultinomialKLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p_proba, q_proba): # [B, T, V]
        # mask = cuda_(torch.zeros(p_proba.size(0), p_proba.size(1)))
        # for i in range(q_proba.size(0)):
        #     for j in range(q_proba.size(1)):
        #         topv, topi = torch.max(p_proba[i,j], -1)
        #         if topi.item() == 0:
        #             mask[i,j] = 0
        #         else:
        #             mask[i,j] = 1
        loss = q_proba * (torch.log(q_proba) - torch.log(p_proba))
        # masked_loss = torch.sum(mask.unsqueeze(-1) * loss, dim=-1)
        # return masked_loss.mean()
        return torch.sum(loss, dim=-1).mean()

class MultinomialKLDivergenceLoss_Corr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p_proba, q_proba): # [B, T, V]
        loss = q_proba * (torch.log(q_proba) - torch.log(p_proba))
        loss = torch.sum(loss, dim=2)   # sum over vocabulary
        loss = torch.sum(loss, dim=1)   # sum over sequence
        return loss.mean()


class SemiCVAE(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(has_qnet=True, **kwargs)
        self.kl_loss = MultinomialKLDivergenceLoss()
        self.bleu_scorer = BLEUScorer()
        # self.kl_loss = MultinomialKLDivergenceLoss_Corr()


    def forward(self, u_input, m_input, z_input, a_input, turn_states, z_supervised,
                        mode, db_vec=None, filling_vec=None, no_label_train=False):
        if mode in ['train', 'loss_eval', 'rl_tune']:
            z_input = None if not z_supervised else z_input
            probs, index, turn_states = \
                self.forward_turn(u_input=u_input, m_input=m_input,
                                            z_input=z_input, a_input=a_input,
                                            turn_states=turn_states, db_vec=db_vec,
                                            filling_vec=filling_vec, is_train=True, mode=mode)
            if z_supervised and mode != 'rl_tune':
                z_input = torch.cat(list(z_input.values()), dim=1)
                a_input = torch.cat(list(a_input.values()), dim=1) if cfg.model_act else None
                index.update({'z_input': z_input, 'a_input': a_input, 'm_input': m_input})
                loss, pz_loss, qz_loss, pa_loss, qa_loss, m_loss = self.supervised_loss(probs, index, no_label_train)
                losses = {'loss': loss, 'pz_loss': pz_loss, 'qz_loss': qz_loss, 'm_loss': m_loss}
                if cfg.model_act:
                    losses.update({'pa_loss': pa_loss, 'qa_loss': qa_loss})
                return losses, turn_states
            elif mode != 'rl_tune':
                index.update({'m_input': m_input})
                if not no_label_train:
                    loss, kl_loss, kl_a_loss, m_loss = self.unsupervised_loss(probs, index)
                    losses = {'loss': loss, 'kl_loss': kl_loss, 'm_loss': m_loss}
                    if cfg.model_act:
                        losses.update({'kl_a_loss': kl_a_loss})
                else:
                    loss, pz_loss, qz_loss, pa_loss, qa_loss, m_loss = self.supervised_loss(probs, index, no_label_train)
                    losses = {'loss': loss, 'm_loss': m_loss}
            else:
                index.update({'m_input': m_input})
                loss, reward= self.rl_loss(probs, index)
                losses = {'loss': loss, 'reward': reward}
            return losses, turn_states

        elif mode == 'test':
            index, db, turn_states = self.forward_turn(u_input=u_input,
                        is_train=False, turn_states=turn_states, db_vec=db_vec)
            return index, db, turn_states


    def forward_turn(self, u_input, turn_states, is_train, m_input=None, z_input=None,
                                a_input=None, db_vec=None, filling_vec=None, mode=None):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_len:
        :param turn_states:
        :param u_input: [B,T]
        :param m_input: [B,T]
        :param z_input: [B,T]
        pv_pz_pr: K * [B,T,V]
        pv_z_dec_outs: K * [B,T,H]
        :return:
        """

        batch_size = u_input.size(0)
        u_hiddens, u_last_hidden = self.u_encoder(u_input)
        u_input_1hot = get_one_hot_input(u_input, self.vocab_size)

        if is_train:
            u_hiddens_q, u_last_hidden_q = self.u_encoder_q(u_input)
            m_hiddens, m_last_hidden = self.m_encoder(m_input)
            m_input_1hot = get_one_hot_input(m_input, self.vocab_size)
            # Q(z|pv_z, u, m)
            sample_type = cfg.sample_type if z_input is None else 'supervised'
            qz_prob, qz_samples, qz_hiddens, turn_states, _ = \
                self.decode_z(batch_size, u_input, u_hiddens_q, u_input_1hot, u_last_hidden_q,
                                    z_input,turn_states, m_input=m_input, m_hiddens=m_hiddens,
                                    m_input_1hot=m_input_1hot, sample_type=sample_type,
                                    decoder_type='qz')

            # P(z|pv_z, u)
            sample_type = 'posterior' if z_input is None else 'supervised'
            pz_prob, pz_samples, pz_hiddens, turn_states, log_pz = \
                self.decode_z(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, z_input,
                                        turn_states, qz_samples=qz_samples, qz_hiddens=qz_hiddens,
                                        sample_type=sample_type, decoder_type='pz')

            # DB indicator and slot filling indicator
            if z_input is None:
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

            if self.model_act:
                sample_type = cfg.sample_type if z_input is None else 'supervised'
                qa_prob, qa_samples, qa_hiddens, log_qa = \
                    self.decode_a(batch_size, u_input, u_hiddens_q, u_input_1hot, u_last_hidden, a_input,
                                db_vec, filling_vec, m_input=m_input, m_hiddens=m_hiddens,
                                m_input_1hot=m_input_1hot, sample_type=sample_type, decoder_type='qa')

                sample_type = 'posterior' if z_input is None else 'supervised'
                pa_prob, pa_samples, pa_hiddens, log_pa = \
                    self.decode_a(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, a_input,
                                            db_vec, filling_vec, qa_samples=qa_samples, qa_hiddens=qa_hiddens,
                                            sample_type=sample_type, decoder_type='pa')
            else:
                pa_prob, pa_samples, pa_hiddens, qa_prob = None, None, None, None

            pm_prob, m_idx, log_pm = \
                    self.decode_m(batch_size, u_last_hidden, u_input, u_hiddens, u_input_1hot,
                                pz_samples, pz_prob, pz_hiddens, pa_samples, pa_prob, pa_hiddens,
                                db_vec, m_input, is_train=True)
            probs = {'pz_prob': pz_prob, 'pm_prob': pm_prob, 'pa_prob': pa_prob,
                            'qz_prob': qz_prob,  'qa_prob': qa_prob, 'log_pm': log_pm}
            index = {'z_input': pz_samples, 'a_input': pa_samples, 'm_idx': m_idx,}
            return probs, index, turn_states

        else: # testing
            sample_type = 'top1'
            pz_prob, pz_samples, pz_hiddens, turn_states, log_pz = \
                self.decode_z(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, z_input,
                                        turn_states, sample_type=sample_type, decoder_type='pz')

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

            if self.model_act:
                pa_prob, pa_samples, pa_hiddens, log_pa = \
                    self.decode_a(batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, a_input,
                                            db_vec, filling_vec, sample_type=sample_type, decoder_type='pa')
            else:
                pa_prob, pa_samples, pa_hiddens = None, None, None

            if not self.beam_search:
                pm_prob, m_idx, log_pm = \
                        self.decode_m(batch_size, u_last_hidden, u_input, u_hiddens, u_input_1hot,
                                    pz_samples, pz_prob, pz_hiddens, pa_samples, pa_prob, pa_hiddens,
                                    db_vec, m_input, is_train=False)
            else:
                m_idx = self.beam_search_decode(u_input, u_input_1hot, u_hiddens, pz_samples,
                                                                    pz_prob, pz_hiddens, db_vec, u_last_hidden[:-1],
                                                                    pa_samples, pa_prob, pa_hiddens)
            z_idx = self.max_sampling(pz_prob)
            a_idx = self.max_sampling(pa_prob) if self.model_act else None
            index = {'m_idx': m_idx, 'z_idx': z_idx, 'a_idx': a_idx}
            return index, match, turn_states


    def supervised_loss(self, probs, index, no_label_train=False):
        pz_prob, qz_prob = torch.log(probs['pz_prob']), torch.log(probs['qz_prob'])
        pm_prob = torch.log(probs['pm_prob'])
        z_input, m_input = index['z_input'], index['m_input']
        pz_loss = self.nll_loss(pz_prob.view(-1, pz_prob.size(2)), z_input.view(-1))
        qz_loss = self.nll_loss(qz_prob.view(-1, qz_prob.size(2)), z_input.view(-1))
        m_loss = self.nll_loss(pm_prob.view(-1, pm_prob.size(2)), m_input.view(-1))
        if self.model_act:
            pa_prob, qa_prob = torch.log(probs['pa_prob']), torch.log(probs['qa_prob'])
            a_input = index['a_input']
            pa_loss = self.nll_loss(pa_prob.view(-1, pa_prob.size(2)), a_input.view(-1))
            qa_loss = self.nll_loss(qa_prob.view(-1, qa_prob.size(2)), a_input.view(-1))
            loss = pz_loss + qz_loss + m_loss + pa_loss + qa_loss
        else:
            pa_loss, qa_loss = torch.zeros(1), torch.zeros(1)
            loss = pz_loss + qz_loss + m_loss
        if no_label_train:
            loss = m_loss
        return loss, pz_loss, qz_loss, pa_loss, qa_loss, m_loss

    def unsupervised_loss(self, probs, index):
        # z_input only used for nll evaluation
        pm_prob = torch.log(probs['pm_prob'])
        m_input = index['m_input']
        m_loss = self.nll_loss(pm_prob.view(-1, pm_prob.size(2)), m_input.view(-1))
        pz_prob, qz_prob = probs['pz_prob'], probs['qz_prob']
        kl_loss = self.kl_loss(pz_prob, qz_prob) * cfg.kl_loss_weight
        if self.model_act:
            pa_prob, qa_prob = probs['pa_prob'], probs['qa_prob']
            kl_a_loss =self.kl_loss(pa_prob, qa_prob) * cfg.kl_loss_weight
            loss = m_loss + kl_loss + kl_a_loss
        else:
            kl_a_loss = torch.zeros(1)
            loss = m_loss + kl_loss
        return loss, kl_loss, kl_a_loss, m_loss


    def rl_loss(self, probs, index):
        """
        :param probs: dict of decoding probabilities, size [B, T, V]
        :param index: dict of decoding indexes, size [B, T]
        """
        def request_score(gen, truth):
            tp, fp, fn = 0, 0, 0
            truth_req, gen_req = set(), set()
            for w in gen.split():
                if '[value_' in w and w.endswith(']') and w != '[value_name]':
                    gen_req.add(w[1:-1].split('_')[1])
            for w in truth.split():
                if '[value_' in w and w.endswith(']') and w != '[value_name]':
                    truth_req.add(w[1:-1].split('_')[1])
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
            precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
            # print('precision:', precision, 'recall:', recall)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return f1

        batch_size = index['m_input'].size()[0]
        # log_prob = probs['log_pa']
        log_prob = probs['log_pm']
        m_true = index['m_input']
        m_gen = index['m_idx']

        loss = 0
        total_reward = 0
        for b in range(batch_size):
            truth = self.reader.vocab.sentence_decode(m_true[b], eos='<eos_r>')
            gen= self.reader.vocab.sentence_decode(m_gen[b], eos='<eos_r>')
            bleu = self.bleu_scorer.score([([gen], [truth])])
            f1 = request_score(gen, truth)
            # f1 = f1 if f1>0.5 else 0
            # print('bleu', bleu, 'f1', f1)
            reward = cfg.rl_coef * bleu + f1
            # reward = f1
            loss += - reward * log_prob[b]
            total_reward += reward
        loss /= batch_size
        total_reward /= batch_size
        return loss, total_reward
