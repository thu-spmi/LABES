import torch
from torch import nn
from utils import BeamState, toss_
from copy_modules import *

class BaseModel(nn.Module):
    def __init__(self, cfg, reader, has_qnet):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.z_length = cfg.z_length
        self.a_length = cfg.a_length
        self.max_len = cfg.m_max_len
        self.teacher_force = cfg.teacher_force
        self.model_act = cfg.model_act
        self.use_act_slot_decoder = cfg.use_act_slot_decoder
        self.multi_domain = False if cfg.dataset == 'camrest' else True
        self.gumbel_temp = cfg.gumbel_temp
        self.prev_z_continuous = cfg.prev_z_continuous
        self.beam_search = cfg.beam_search

        self.reader = reader
        self.vocab = self.reader.vocab
        self.db_op = self.reader.dataset
        self.z_eos_map = self.reader.otlg.z_eos_map
        self.eos_m_token = self.reader.otlg.eos_syntax['resp']

        if self.beam_search:
            self.beam_size = cfg.beam_params['size']
            self.beam_len_bonus =  cfg.beam_params['len_bonus']

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.u_encoder = Encoder(self.embedding, cfg.vocab_size, cfg.embed_size, cfg.hidden_size,
                                                  cfg.layer_num, cfg.dropout_rate)
        self.z_encoder = Encoder(self.embedding, cfg.vocab_size, cfg.embed_size, cfg.hidden_size,
                                                  cfg.layer_num, cfg.dropout_rate)
        self.pz_decoder = PriorDecoder_Pz(cfg.embed_size, cfg.hidden_size, cfg.vocab_size,
                                                                 cfg.dropout_rate, enable_selc_read=False)  # prior
        self.decode_z = self.decode_z_order if not cfg.parallel_z else self.decode_z_parallel

        # if self.cfg.multi_domain:
        #     self.domain_classifier = DomainClassifier(cfg.hidden_size, 8, cfg.dropout_rate)

        if self.model_act:
            self.a_encoder = Encoder(self.embedding, cfg.vocab_size, cfg.embed_size, cfg.hidden_size,
                                                      cfg.layer_num, cfg.dropout_rate)

            self.pa_decoder_v = PriorDecoder_Pa(cfg.embed_size, cfg.hidden_size, cfg.vocab_size,
                                                                        cfg.db_vec_size, len(self.z_eos_map),  cfg.dropout_rate)  # prior
            if self.use_act_slot_decoder and not cfg.share_act_decoder:
                self.pa_decoder_s = PriorDecoder_Pa(cfg.embed_size, cfg.hidden_size, cfg.vocab_size,
                                                                            cfg.db_vec_size, len(self.z_eos_map), cfg.dropout_rate)  # prior
            else:
                self.pa_decoder_s = self.pa_decoder_v
            self.pa_decoder = {'av': self.pa_decoder_v, 'as': self.pa_decoder_s}

        if has_qnet:
            if not cfg.share_q_encoder:
                self.u_encoder_q = Encoder(self.embedding, cfg.vocab_size, cfg.embed_size, cfg.hidden_size,
                                                             cfg.layer_num, cfg.dropout_rate)
            else:
                self.u_encoder_q = self.u_encoder
            self.m_encoder = Encoder(self.embedding, cfg.vocab_size, cfg.embed_size, cfg.hidden_size,
                                                       cfg.layer_num, cfg.dropout_rate)
            self.qz_decoder = PosteriorDecoder_Qz(cfg.embed_size, cfg.hidden_size, cfg.vocab_size,
                                                                             cfg.dropout_rate,
                                                                             enable_selc_read=False)  # posterior
            if self.model_act:
                self.qa_decoder_v = PosteriorDecoder_Qa(cfg.embed_size, cfg.hidden_size, cfg.vocab_size,
                                                                                    cfg.db_vec_size, len(self.z_eos_map),
                                                                                     cfg.dropout_rate,
                                                                                     enable_selc_read=False)  # posterior
                if not cfg.share_act_decoder:
                    self.qa_decoder_s = PosteriorDecoder_Qa(cfg.embed_size, cfg.hidden_size, cfg.vocab_size,
                                                                                        cfg.db_vec_size, len(self.z_eos_map),
                                                                                        cfg.dropout_rate,
                                                                                        enable_selc_read=False)  # posterior
                else:
                    self.qa_decoder_s = self.qa_decoder_v
                self.qa_decoder = {'av': self.qa_decoder_v, 'as': self.qa_decoder_s}

        self.m_decoder = ResponseDecoder_Pm(self.embedding, cfg.embed_size, cfg.hidden_size,
                                                                          cfg.vocab_size, cfg.db_vec_size,
                                                                          cfg.dropout_rate, enable_selc_read=False)
        self.nll_loss = nn.NLLLoss(ignore_index=0)

    def decode_z_order(self, batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, z_input,
                          turn_states, sample_type, decoder_type, qz_samples=None, qz_hiddens=None,
                          m_input=None, m_hiddens=None, m_input_1hot=None, mask_otlg=False):
        return_gmb = True if 'gumbel' in sample_type else False
        pv_z_pr = turn_states.get('pv_%s_pr'%decoder_type, None)
        pv_z_h = turn_states.get('pv_%s_h'%decoder_type, None)
        pv_z_id = turn_states.get('pv_%s_id'%decoder_type, None)
        z_prob, z_samples, gmb_samples = [], [], []
        log_pz = 0
        for si, sn in enumerate(self.reader.otlg.informable_slots):
            last_hidden = u_last_hidden[:-1]
            # last_hidden = (u_last_hidden[-1] + u_last_hidden[-2]).unsqueeze(0)
            z_eos_idx = self.vocab.encode(self.z_eos_map[sn])
            emb_zt = self.get_first_z_input(sn, batch_size, self.multi_domain)
            zero_vec = cuda_(torch.zeros(batch_size, 1, self.hidden_size))
            selc_read_u = selc_read_m = selc_read_pv_z = zero_vec
            if pv_z_pr is not None:
                b, e = si * self.z_length, (si+1) * self.z_length
                pv_pr, pv_h, pv_idx = pv_z_pr[:, b:e], pv_z_h[:, b:e], pv_z_id[:, b:e]
            else:
                pv_pr, pv_h, pv_idx = None, None, None
            prev_zt = None
            for t in range(self.z_length):
                if decoder_type == 'pz':
                    prob, last_hidden, gru_out, selc_read_u, selc_read_pv_z = \
                        self.pz_decoder(u_input, u_input_1hot, u_hiddens,
                                            pv_z_prob=pv_pr, pv_z_hidden=pv_h, pv_z_idx=pv_idx,
                                            emb_zt=emb_zt,  last_hidden=last_hidden,
                                            selc_read_u=selc_read_u, selc_read_pv_z=selc_read_pv_z)
                else:
                    prob, last_hidden, gru_out, selc_read_u, selc_read_m, selc_read_pv_z, gmb_samp = \
                        self.qz_decoder(u_input, u_input_1hot, u_hiddens, m_input, m_input_1hot, m_hiddens,
                                                pv_z_prob=pv_pr, pv_z_hidden=pv_h, pv_z_idx=pv_idx,
                                                emb_zt=emb_zt, last_hidden=last_hidden, selc_read_u=selc_read_u,
                                                selc_read_m=selc_read_m, selc_read_pv_z=selc_read_pv_z,
                                                temp=self.gumbel_temp, return_gmb=return_gmb)
                if mask_otlg:
                    prob = self.mask_probs(prob, tokens_allow=self.reader.slot_value_mask[sn])

                if sample_type == 'supervised':
                    zt = z_input[sn][:, t]
                elif sample_type == 'top1':
                    zt = torch.topk(prob, 1)[1]
                elif sample_type == 'topk':
                    topk_probs, topk_words = torch.topk(prob.squeeze(1), cfg.topk_num)
                    widx = torch.multinomial(topk_probs, 1, replacement=True)
                    zt = torch.gather(topk_words, 1, widx)      #[B]
                elif sample_type == 'posterior':
                    zt = qz_samples[:, si * self.z_length + t]
                elif 'gumbel' in sample_type:
                    zt = torch.argmax(gmb_samp, dim=1)   #[B]
                    emb_zt = torch.matmul(gmb_samp, self.embedding.weight).unsqueeze(1) # [B, 1, H]
                    zt, prev_zt, gmb_samp = self.mask_samples(zt, prev_zt, batch_size, z_eos_idx, gmb_samp, True)
                    gmb_samples.append(gmb_samp)

                if 'gumbel' not in sample_type:
                    emb_zt = self.embedding(zt.view(-1, 1))
                    prob_zt = torch.gather(prob, 1, zt.view(-1, 1)).squeeze(1) #[B, 1]
                    log_prob_zt = torch.log(prob_zt)
                    zt, prev_zt, log_prob_zt = self.mask_samples(zt, prev_zt, batch_size, z_eos_idx, log_prob_zt)
                    log_pz += log_prob_zt
                z_samples.append(zt.view(-1))
                z_prob.append(prob)

        z_prob = torch.stack(z_prob, dim=1)  # [B*K,Tz,V]
        z_samples= torch.stack(z_samples, dim=1)  # [B*K,Tz]
        if sample_type == 'posterior':
            z_samples, z_hiddens = qz_samples, qz_hiddens
        elif 'gumbel' not in sample_type:
            z_hiddens, z_last_hidden = self.z_encoder(z_samples, input_type='index')
        else:
            z_gumbel = torch.stack(gmb_samples, dim=1)   # [B,Tz, V]
            z_gumbel = torch.matmul(z_gumbel, self.embedding.weight)     # [B,Tz, E]
            z_hiddens, z_last_hidden = self.z_encoder(z_gumbel, input_type='embedding')

        retain = self.prev_z_continuous
        turn_states['pv_%s_h'%decoder_type] = z_hiddens if retain else z_hiddens.detach()
        turn_states['pv_%s_pr'%decoder_type] = z_prob if retain else z_prob.detach()
        turn_states['pv_%s_id'%decoder_type] = z_samples if retain else z_samples.detach()

        return z_prob, z_samples, z_hiddens, turn_states, log_pz


    def decode_z_parallel(self, batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, z_input,
                          turn_states, sample_type, decoder_type, qz_samples=None, qz_hiddens=None,
                          m_input=None, m_hiddens=None, m_input_1hot=None, mask_otlg=False):
        return_gmb = True if 'gumbel' in sample_type else False

        slot_num = len(self.reader.otlg.informable_slots)
        pv_z_pr = turn_states.get('pv_%s_pr'%decoder_type, None)
        pv_z_h = turn_states.get('pv_%s_h'%decoder_type, None)
        pv_z_id = turn_states.get('pv_%s_id'%decoder_type, None)
        z_prob, z_samples, gmb_samples = [], [], []
        log_pz = 0

        last_hidden = u_last_hidden[:-1].repeat(1, slot_num, 1)   # [1, B*|slot|, H]
        u_input = u_input.repeat(slot_num ,1)
        u_input_1hot = u_input_1hot.repeat(slot_num, 1, 1)
        # u_input_1hot = get_one_hot_input(u_input, self.vocab_size)

        u_hiddens = u_hiddens.repeat(slot_num ,1, 1)
        if decoder_type != 'pz':
            m_input = m_input.repeat(slot_num ,1)
            m_input_1hot = m_input_1hot.repeat(slot_num, 1, 1)
            m_hiddens = m_hiddens.repeat(slot_num ,1, 1)
            # m_input_1hot = get_one_hot_input(m_input, self.vocab_size)
        if pv_z_pr is not None:
            # pv_z_pr = pv_z_pr.transpose(1,0).reshape(self.z_length, -1, cfg.vocab_size).transpose(1,0)    # [B*|slot|, T, V]
            # pv_z_h = pv_z_h.transpose(1,0).reshape(self.z_length, -1, cfg.hidden_size).transpose(1,0)    # [B*|slot|, T, H]
            # pv_z_id = pv_z_id.transpose(1,0).reshape(self.z_length, -1).transpose(1,0)    # [B*|slot|, T]
            pv_z_prob, pv_z_hid, pv_z_idx = [], [], []
            for si, sn in enumerate(self.reader.otlg.informable_slots):
                pv_z_prob.append(pv_z_pr[:, si*self.z_length : (si+1)*self.z_length])
                pv_z_hid.append(pv_z_h[:, si*self.z_length : (si+1)*self.z_length])
                pv_z_idx.append(pv_z_id[:, si*self.z_length : (si+1)*self.z_length])
            pv_z_pr = torch.cat(pv_z_prob, dim=0)
            pv_z_h = torch.cat(pv_z_hid, dim=0)
            pv_z_id = torch.cat(pv_z_idx, dim=0)
            # print(pv_z_pr.size())
            # print(pv_z_id)

        emb_zt, z_eos_idx = [], []
        for si, sn in enumerate(self.reader.otlg.informable_slots):
            z_eos_idx.append(self.vocab.encode(self.z_eos_map[sn]))
            emb_zt.append(self.get_first_z_input(sn, batch_size, self.multi_domain))
        emb_zt = torch.cat(emb_zt, dim=0)

        if z_input is not None:
            z_input_cat = []
            for si, sn in enumerate(self.reader.otlg.informable_slots):
                z_input_cat.append(z_input[sn])
            z_input_cat = torch.cat(z_input_cat, dim=0)

        zero_vec = cuda_(torch.zeros(batch_size * slot_num, 1, self.hidden_size))
        selc_read_u = selc_read_m = selc_read_pv_z = zero_vec

        prev_zt = None
        for t in range(self.z_length):
            if decoder_type == 'pz':
                prob, last_hidden, gru_out, selc_read_u, selc_read_pv_z = \
                    self.pz_decoder(u_input, u_input_1hot, u_hiddens,
                                        pv_z_prob=pv_z_pr, pv_z_hidden=pv_z_h, pv_z_idx=pv_z_id,
                                        emb_zt=emb_zt,  last_hidden=last_hidden,
                                        selc_read_u=selc_read_u, selc_read_pv_z=selc_read_pv_z)
            else:

                prob, last_hidden, gru_out, selc_read_u, selc_read_m, selc_read_pv_z, gmb_samp = \
                    self.qz_decoder(u_input, u_input_1hot, u_hiddens, m_input, m_input_1hot, m_hiddens,
                                            pv_z_prob=pv_z_pr, pv_z_hidden=pv_z_h, pv_z_idx=pv_z_id,
                                            emb_zt=emb_zt, last_hidden=last_hidden, selc_read_u=selc_read_u,
                                            selc_read_m=selc_read_m, selc_read_pv_z=selc_read_pv_z,
                                            temp=self.gumbel_temp, return_gmb=return_gmb)
            if mask_otlg:
                prob = self.mask_probs(prob, tokens_allow=self.reader.slot_value_mask[sn])

            if sample_type == 'supervised':
                # zt = z_input[sn][:, t]
                zt = z_input_cat[:, t]
            elif sample_type == 'top1':
                zt = torch.topk(prob, 1)[1]
            elif sample_type == 'topk':
                topk_probs, topk_words = torch.topk(prob.squeeze(1), cfg.topk_num)
                widx = torch.multinomial(topk_probs, 1, replacement=True)
                zt = torch.gather(topk_words, 1, widx)      #[B]
            elif sample_type == 'posterior':
                zt = qz_samples[:, si * self.z_length + t]
            elif 'gumbel' in sample_type:
                zt = torch.argmax(gmb_samp, dim=1)   #[B]
                emb_zt = torch.matmul(gmb_samp, self.embedding.weight).unsqueeze(1) # [B, 1, H]
                zt, prev_zt, gmb_samp = self.mask_samples(zt, prev_zt, batch_size, z_eos_idx, gmb_samp, True)
                gmb_samples.append(gmb_samp)

            if 'gumbel' not in sample_type:
                emb_zt = self.embedding(zt.view(-1, 1))
                prob_zt = torch.gather(prob, 1, zt.view(-1, 1)).squeeze(1) #[B, 1]
                log_prob_zt = torch.log(prob_zt)
                zt, prev_zt, log_prob_zt = self.mask_samples(zt, prev_zt, batch_size, z_eos_idx, log_prob_zt)
                log_pz += log_prob_zt
                z_samples.append(zt.view(-1))
                z_prob.append(prob)

        z_prob = torch.stack(z_prob, dim=1)  # [B*|slot|,Tz,V]
        z_samples= torch.stack(z_samples, dim=1)  # [B*|slot|,Tz]
        z_prob_col, z_samples_col = [], []
        for i in range(slot_num):
            z_prob_col.append(z_prob[i*batch_size : (i+1)*batch_size])
            z_samples_col.append(z_samples[i*batch_size : (i+1)*batch_size])
        z_prob = torch.cat(z_prob_col, dim=1)  # [B,Tz*|slot|,V]
        z_samples= torch.cat(z_samples_col, dim=1)  # [B,Tz*|slot|]

        #    Tz*|slot|, B

        if sample_type == 'posterior':
            z_samples, z_hiddens = qz_samples, qz_hiddens
        elif 'gumbel' not in sample_type:
            z_hiddens, z_last_hidden = self.z_encoder(z_samples, input_type='index')
        else:
            z_gumbel = torch.stack(gmb_samples, dim=1)   # [B, Tz, V]
            z_gumbel = torch.matmul(z_gumbel, self.embedding.weight)     # [B, Tz, E]
            z_hiddens, z_last_hidden = self.z_encoder(z_gumbel, input_type='embedding')

        retain = self.prev_z_continuous
        turn_states['pv_%s_h'%decoder_type] = z_hiddens if retain else z_hiddens.detach()
        turn_states['pv_%s_pr'%decoder_type] = z_prob if retain else z_prob.detach()
        turn_states['pv_%s_id'%decoder_type] = z_samples if retain else z_samples.detach()

        return z_prob, z_samples, z_hiddens, turn_states, log_pz


    def decode_a(self, batch_size, u_input, u_hiddens, u_input_1hot, u_last_hidden, a_input,
                          db_vec, filling_vec, sample_type, decoder_type, qa_samples=None,
                           qa_hiddens=None, m_input=None, m_hiddens=None, m_input_1hot=None):
        return_gmb = True if 'gumbel' in sample_type else False
        a_prob, a_samples, gmb_samples = [], [], []
        log_pa = 0
        for si, sn in enumerate(self.reader.act_order):
            last_hidden = u_last_hidden[:-1]
            a_eos_idx = self.vocab.encode('<eos_%s>'%sn)
            a_sos_idx = self.vocab.encode('<go_%s>'%sn)
            at = cuda_(torch.ones(batch_size, 1)*a_sos_idx).long()
            emb_at = self.embedding(at)
            selc_read_m = cuda_(torch.zeros(batch_size, 1, self.hidden_size))
            vec_input = torch.cat([db_vec, filling_vec], dim=1)
            if sn == 'av':
                vec_input = cuda_(torch.zeros(vec_input.size()))
            prev_at = None
            for t in range(self.a_length):
                if decoder_type == 'pa':
                    prob, last_hidden, gru_out = self.pa_decoder[sn](
                                u_hiddens, emb_at, vec_input, last_hidden)
                else:
                    prob, last_hidden, gru_out, selc_read_m, gmb_samp = \
                        self.qa_decoder[sn](u_hiddens,
                                                        m_input, m_input_1hot, m_hiddens,
                                                        emb_at, vec_input, last_hidden,
                                                        selc_read_m=selc_read_m, temp=self.gumbel_temp,
                                                        return_gmb=return_gmb)
                if sample_type == 'supervised':
                    at = a_input[sn][:, t]
                elif sample_type == 'top1':
                    at = torch.topk(prob, 1)[1]
                elif sample_type == 'topk':
                    topk_probs, topk_words = torch.topk(prob.squeeze(1), cfg.topk_num)
                    widx = torch.multinomial(topk_probs, 1, replacement=True)
                    at = torch.gather(topk_words, 1, widx)      #[B]
                elif sample_type == 'posterior':
                    at = qa_samples[:, si * self.a_length + t]
                elif 'gumbel' in sample_type:
                    at = torch.argmax(gmb_samp, dim=1)   #[B]
                    emb_at = torch.matmul(gmb_samp, self.embedding.weight).unsqueeze(1) # [B, 1, H]
                    at, prev_at, gmb_samp = self.mask_samples(at, prev_at, batch_size, a_eos_idx, gmb_samp, True)
                    gmb_samples.append(gmb_samp)

                if 'gumbel' not in sample_type:
                    emb_at = self.embedding(at.view(-1, 1))
                    prob_at = torch.gather(prob, 1, at.view(-1, 1)).squeeze(1) #[B, 1]
                    log_prob_at = torch.log(prob_at)
                    at, prev_at, log_prob_at = self.mask_samples(at, prev_at, batch_size, a_eos_idx, log_prob_at)
                    log_pa += log_prob_at
                a_samples.append(at.view(-1))
                a_prob.append(prob)
        a_prob = torch.stack(a_prob, dim=1)
        a_samples = torch.stack(a_samples, dim=1)  # [B,Ta]

        if sample_type == 'posterior':
            a_samples, a_hiddens = qa_samples, qa_hiddens
        elif 'gumbel' not in sample_type:
            a_hiddens, a_last_hidden = self.a_encoder(a_samples, input_type='index')
        else:
            a_gumbel = torch.stack(gmb_samples, dim=1)   # [B,Ta, V]
            a_gumbel = torch.matmul(a_gumbel, self.embedding.weight)     # [B,Ta, E]
            a_hiddens, a_last_hidden = self.a_encoder(a_gumbel, input_type='embedding')

        return a_prob, a_samples, a_hiddens, log_pa

    def decode_m(self, batch_size, u_last_hidden, u_input, u_hiddens, u_input_1hot,
                            pz_samples, pz_prob, z_hiddens, pa_samples, pa_prob, a_hiddens,
                            db_vec, m_input, is_train):
        last_hidden = u_last_hidden[:-1]
        mt = cuda_(torch.ones(batch_size, 1).long())  # GO token
        m_idx, pm_prob = [], []
        zero_vec = cuda_(torch.zeros(batch_size, 1, self.hidden_size))
        selc_read_u = selc_read_z = selc_read_a = zero_vec
        m_eos_idx = self.vocab.encode(self.eos_m_token)
        log_pm = 0
        prev_mt = None
        m_len = m_input.size(1) if m_input is not None else self.max_len
        for t in range(m_len):
            prob, last_hidden, gru_out, selc_read_u, selc_read_z, selc_read_a = \
            self.m_decoder(u_input, u_input_1hot, u_hiddens,
                                     pz_samples, pz_prob, z_hiddens,
                                     mt, db_vec, last_hidden,
                                     pa_samples, pa_prob, a_hiddens,
                                     selc_read_u, selc_read_z, selc_read_a)
            if is_train:
                teacher_forcing = toss_(self.teacher_force)
                mt = m_input[:, t] if teacher_forcing else torch.topk(prob, 1)[1]
                mt = mt.view(-1, 1)
                prob_mt = torch.gather(prob, 1, mt).squeeze(1) #[B*K]
                log_pmt = torch.log(prob_mt)
                mt, prev_mt, log_pmt = self.mask_samples(mt, prev_mt, batch_size, m_eos_idx, log_pmt)
                log_pm += log_pmt
                m_idx.append(torch.topk(prob, 1)[1].data.view(-1))
            else:
                mt_prob, mt = torch.topk(prob, 1)
                m_idx.append(mt.data.view(-1))
            pm_prob.append(prob)
        pm_prob = torch.stack(pm_prob, dim=1)  # [B,T,V]
        m_idx = torch.stack(m_idx, dim=1)
        m_idx = [list(_) for _ in list(m_idx)]
        return pm_prob, m_idx, log_pm

    def max_sampling(self, prob):
        """
        Max-sampling procedure of pz during testing.
        :param pz_prob: # [B, T, V]
        :return: nested-list: B * [T]
        """
        p, token = torch.topk(prob, 1, dim=2)  # [B, T, 1]
        token = list(token.squeeze(2))
        return [list(_) for _ in token]


    def get_first_z_input(self, domain_slot_name, batch_size, multi_domain=True):
        # domain_slot_name: a string likes "domain-slot"

        domain, slot = domain_slot_name.split('-')
        smap = { 'pricerange': 'price',
                        'weather_attribute': 'weather',
                        'poi_type': 'type',
                        'traffic_info':  'traffic',
                        }
        slot = smap.get(slot, slot)
        slot_idx = self.vocab.encode(slot)
        slot_idx = cuda_(torch.ones(batch_size, 1)*slot_idx).long()
        emb = self.embedding(slot_idx)
        if multi_domain:
            domain_idx = self.vocab.encode(domain)
            domain_idx = cuda_(torch.ones(batch_size, 1)*domain_idx).long()
            d_emb = self.embedding(domain_idx)
            emb = emb + d_emb
        return emb


    def mask_samples(self, curr_z, prev_z, batch_size, eos_idx, curr_z_pr=None, is_gumbel=False):
        # change tokens and probabilities after <eos> to zeros
        # necessary for log-likelihood computation in SMC
        if not isinstance(eos_idx, list):
            eos_idx = [eos_idx]
        mask = cuda_(torch.ones(curr_z.size()))
        if prev_z is not None:
            for b in range(batch_size):
                if prev_z[b].item() in eos_idx or prev_z[b].item() == 0:
                    mask[b] = 0
        curr_z_masked = curr_z * mask.long()
        new_prev_z = curr_z_masked
        if curr_z_pr is not None:
            if not is_gumbel:
                curr_z_pr_masked = curr_z_pr * mask.view(curr_z_pr.size())
            else:
                mask = mask.view(-1, 1)
                curr_z_pr_masked = curr_z_pr * mask.expand(mask.size(0), curr_z_pr.size(1))
            return curr_z_masked, new_prev_z, curr_z_pr_masked
        else:
            return curr_z_masked, new_prev_z



    def mask_probs(self, prob, tokens_allow):
        """ set probability of all the special tokens to ~0 except tokens in list special_allow
        : param prob: size [B,V]
        :param special_allow: [description]
        :type special_allow: [type]
        """
        # mask_idx = [0,1,3,4,5,6,7,8]
        # for idx in special_allow:
        #     if idx in mask_idx:
        #         mask_idx.remove(idx)
        # mask = cuda_(torch.ones(prob.size()))
        # mask[:, mask_idx] = 1e-10
        mask = cuda_(torch.zeros(prob.size()).fill_(1e-10))
        mask[:, tokens_allow] = 1
        prob = prob * mask
        return prob


    def beam_search_decode(self, u_input, u_input_1hot, u_hiddens, pz_samples, pz_prob, z_hiddens,
                                            db_vec, last_hidden, pa_samples=None, pa_prob=None, a_hiddens=None):
        # Beam search decoding does not support selective read
        batch_size = u_input.size(0)
        decoded = []
        for b in range(batch_size):
            u_input_s = u_input[b].unsqueeze(0)
            u_input_1hot_s = u_input_1hot[b].unsqueeze(0)
            u_hiddens_s = u_hiddens[b].unsqueeze(0)
            pz_samples_s = pz_samples[b].unsqueeze(0)
            pz_prob_s = pz_prob[b].unsqueeze(0)
            z_hiddens_s = z_hiddens[b].unsqueeze(0)
            db_vec_s = db_vec[b].unsqueeze(0)
            last_hidden_s = last_hidden[:, b].unsqueeze(1)
            if pa_samples is not None:
                pa_samples_s = pa_samples[b].unsqueeze(0)
                pa_prob_s = pa_prob[b].unsqueeze(0)
                a_hiddens_s = a_hiddens[b].unsqueeze(0)
            else:
                pa_samples_s, pa_prob_s, a_hiddens_s = None, None, None
            decoded_s = self.beam_single(u_input_s, u_input_1hot_s, u_hiddens_s, pz_samples_s,
                                                             pz_prob_s, z_hiddens_s, db_vec_s, last_hidden_s,
                                                             pa_samples_s, pa_prob_s, a_hiddens_s)
            decoded.append(decoded_s)
        return [list(_.view(-1)) for _ in decoded]

    def beam_single(self, u_input, u_input_1hot, u_hiddens, pz_samples, pz_prob, z_hiddens, db_vec, last_hidden,
                                pa_samples=None, pa_prob=None, a_hiddens=None):
        assert u_input.size(0) == 1, "Beam search single requires batch size to be 1"


        def beam_result_valid(decoded_t):
            return True

        def score_bonus(state, decoded):
            """
            bonus scheme: bonus per token, or per new decoded slot.
            :param state:
            :return:
            """
            bonus = self.beam_len_bonus
            # decoded = self.vocab.decode(decoded)
            # decoded_t = [_.item() for _ in state.decoded]
            # decoded_sentence = self.vocab.sentence_decode(decoded_t, self.eos_m_token)
            # decoded_sentence = decoded_sentence.split()
            if len(state.decoded) >= 1 and state.decoded[-1] == decoded: # repeated words
                # print('repeat words!')
                bonus -= 1000
            if decoded == '**unknown**':
                bonus -= 3.0
            return bonus


        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        mt = cuda_(torch.ones(1, 1).long())  # GO token
        states.append(BeamState(0, last_hidden, [mt], 0))

        for t in range(self.max_len):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, mt = state.last_hidden, state.decoded[-1]
                prob, last_hidden, _, _, _, _ = \
                    self.m_decoder(u_input, u_input_1hot, u_hiddens,
                                             pz_samples, pz_prob, z_hiddens,
                                             mt, db_vec, last_hidden,
                                             pa_samples, pa_prob, a_hiddens)

                prob = torch.log(prob)
                mt_prob, mt_index = torch.topk(prob, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = mt_prob[0][new_k].item() + score_bonus(state,
                                                                                                mt_index[0][new_k].item())
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if decoded_t.item() >= self.vocab_size:
                        decoded_t = cuda_(torch.ones(1, 1)*2).long()  # unk
                    if self.vocab.decode(decoded_t.item()) == self.eos_m_token:
                        if beam_result_valid(state.decoded):
                            finished.append(state)
                            dead_k += 1
                        else:
                            failed.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_len - 1 and not finished:
                finished = failed
                print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).item() for _ in decoded_t]
        # decoded_sentence = self.vocab.sentence_decode(decoded_t, self.eos_m_token)
        # print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def self_adjust(self, epoch):
        pass