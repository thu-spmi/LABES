import torch
import torch.nn.functional as F
from torch import nn
from config import global_config as cfg
from modules import *

def get_selective_read(source, target, hiddens, copy_probs):
        cp_pos = torch.stack([sb==target[b] for b, sb in enumerate(source)], dim=0)  # [B,T]
        weight = copy_probs * cp_pos.float()  # [B,Tu]
        weight.masked_fill_(weight==0, -1e10)
        weight = F.softmax(weight, dim=1)
        selective_read = torch.bmm(weight.unsqueeze(1), hiddens)  # [B,1,H]
        return selective_read


class PriorDecoder_Pz(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate, enable_selc_read=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = hidden_size + embed_size
        self.dropout_rate = dropout_rate
        self.enable_selc_read = enable_selc_read
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, dropout=dropout_rate,
                                      batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        # weights
        if self.enable_selc_read:
            self.Win = nn.Linear(embed_size + hidden_size*2, embed_size)
        self.Wctx = nn.Linear(hidden_size*2, hidden_size)
        # self.Wst = nn.Linear(self.state_size, hidden_size)
        self.Wgen = nn.Linear(self.state_size, vocab_size) # generate mode
        self.Wcp_u = nn.Linear(hidden_size, self.state_size) # copy mode
        self.Wcp_pz = nn.Linear(hidden_size, self.state_size) # copy mode

        self.attn = Attn(hidden_size)
        # self.mu = nn.Linear(vocab_size, embed_size, bias=False)

    def forward(self, u_input, u_input_1hot, u_hiddens, pv_z_prob, pv_z_hidden, pv_z_idx,
                        emb_zt, last_hidden, selc_read_u=None, selc_read_pv_z=None):
        """[summary]
        :param u_input: [B,Tu]
        :param u_input_1hot: [B,Tu,V]
        :param u_hiddens: [B,Tu,H]
        :param pv_pz_prob: [B,Tz,V]
        :param pv_z_hidden: [B,Tz,H]
        :param emb_zt: [B,1,He]
        :param selc_read_u: [B,1,H]
        :param selc_read_pv_z: [B,1,H]
        :param last_hidden: [1,B,H]
        """
        V = u_input_1hot.size(2)
        Tu = u_input.size(1)

        if self.enable_selc_read:
            t_input = self.Win(torch.cat([emb_zt, selc_read_u, selc_read_pv_z], dim=2))   # [B,1,H]
        else:
            t_input = emb_zt

        if pv_z_hidden is not None:
            context = self.attn(last_hidden, torch.cat([u_hiddens, pv_z_hidden], dim=1))
        else:
            context = self.attn(last_hidden, u_hiddens)
        gru_input = torch.cat([t_input, context], dim=2)
        gru_input = self.dropout(gru_input)

        gru_out, last_hidden = self.gru(gru_input, last_hidden)  # gru_out: [B,1,H]

        st = torch.cat([gru_out, t_input], dim=2)   # depends more on slot name
        if cfg.dropout_st:
            st = self.dropout(st)
        score_g = self.Wgen(st).squeeze(1)     # [B,V]

        score_c_u = torch.tanh(self.Wcp_u(u_hiddens))   # [B,Tu,H]
        score_c_u = torch.bmm(score_c_u, st.transpose(1, 2)).squeeze(2)      # [B,Tu]

        if pv_z_prob is None:
            # copy from only user input
            score = torch.cat([score_g, score_c_u],1) # [B, V+Tu]
            probs = F.softmax(score, dim=1)
            prob_g, prob_c_u= probs[:, :V],  probs[:, V:]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g
            selc_read_pz = cuda_(torch.zeros(u_input.size(0), 1, self.hidden_size))
        else:
            # copy from only user input and previous z decoded
            score_c_pz = torch.tanh(self.Wcp_pz(pv_z_hidden))   # [B,Tz,H]
            score_c_pz = torch.bmm(score_c_pz, st.transpose(1, 2)).squeeze(2)      # [B,Tz]
            score = torch.cat([score_g, score_c_u, score_c_pz],1) # [B, V+Tu+Tz]
            probs = F.softmax(score, dim=1)
            prob_g, prob_c_u, prob_c_pz = probs[:, :V],  probs[:, V: V+Tu],  probs[:, V+Tu: ]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_pz_to_g = torch.bmm(prob_c_pz.unsqueeze(1), pv_z_prob).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_pz_to_g  # [B,V]

            # compute selcive read from pv_z for the next step
            if self.enable_selc_read:
                selc_read_pz = get_selective_read(pv_z_idx, zt, pv_z_hidden, prob_c_pz)

        # compute selcive read from u and m for the next step
        if self.enable_selc_read:
            selc_read_u = get_selective_read(u_input, zt, u_hiddens, prob_c_u)
        else:
            selc_read_u, selc_read_pz = None, None

        return prob_out, last_hidden, gru_out, selc_read_u, selc_read_pz


class PosteriorDecoder_Qz(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate,
                       enable_selc_read=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = embed_size + hidden_size
        self.dropout_rate = dropout_rate
        self.enable_selc_read = enable_selc_read
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, dropout=dropout_rate,
                                      batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        # weights
        if self.enable_selc_read:
            self.Win = nn.Linear(embed_size + hidden_size*3, embed_size)
        # self.Wst = nn.Linear(self.state_size, hidden_size)
        self.Wgen = nn.Linear(self.state_size, vocab_size) # generate mode
        self.Wcp_u = nn.Linear(hidden_size, self.state_size) # copy mode
        self.Wcp_m = nn.Linear(hidden_size, self.state_size) # copy mode
        self.Wcp_pz = nn.Linear(hidden_size, self.state_size) # copy mode

        self.attn = Attn(hidden_size)

    def forward(self, u_input, u_input_1hot, u_hiddens, m_input, m_input_1hot, m_hiddens,
                        pv_z_prob, pv_z_hidden, pv_z_idx, emb_zt, last_hidden, selc_read_u=None,
                        selc_read_m=None, selc_read_pv_z=None, temp=0.1, return_gmb=False):
        """[summary]
        :param u_input: [B,Tu]
        :param u_input_1hot: [B,Tu,V]
        :param u_hiddens: [B,Tu,H]
        :param m_input: [B,Tm]
        :param m_input_1hot: [B,Tm,V]
        :param m_hiddens: [B,Tm,H]
        :param pv_pz_prob: [B,Tz,V]
        :param pv_z_hidden: [B,Tz,H]
        :param emb_zt: [B,1,H]
        :param selc_read_u: [B,1,H]
        :param selc_read_pv_z: [B,1,H]
        :param last_hidden: [1,B,H]
        :param temp: gumbel softmax temperature
        :param return_gmb: return gumbel softmax samples
        """
        V = u_input_1hot.size(2)
        Tu, Tm = u_input.size(1), m_input.size(1)

        if self.enable_selc_read:
            t_input = self.Win(torch.cat([emb_zt, selc_read_u, selc_read_m, selc_read_pv_z], dim=2))
        else:
            t_input = emb_zt   # [B,1,H]

        hiddens = [u_hiddens, m_hiddens]
        if pv_z_hidden is not None: hiddens.append(pv_z_hidden)
        context = self.attn(last_hidden, torch.cat(hiddens, dim=1))
        gru_input = torch.cat([t_input, context], dim=2)
        gru_input = self.dropout(gru_input)
        gru_out, last_hidden = self.gru(gru_input, last_hidden)  # gru_out: [B,1,H]
        st = torch.cat([gru_out, t_input], dim=2)
        if cfg.dropout_st: st = self.dropout(st)

        score_g = self.Wgen(st).squeeze(1)     # [B,V]
        score_c_u = torch.tanh(self.Wcp_u(u_hiddens))   # [B,Tu,H]
        score_c_u = torch.bmm(score_c_u, st.transpose(1, 2)).squeeze(2)      # [B,Tu]
        score_c_m = torch.tanh(self.Wcp_m(m_hiddens))   # [B,Tu,H]
        score_c_m = torch.bmm(score_c_m, st.transpose(1, 2)).squeeze(2)      # [B,Tu]

        if pv_z_prob is None:
            # copy from only user input and system response
            score_no_pv = torch.cat([score_g, score_c_u, score_c_m],1) # [B, V+Tu+Tm]
            probs = F.softmax(score_no_pv, dim=1)
            prob_g, prob_c_u, prob_c_m = probs[:, :V], probs[:, V:V+Tu], probs[:, V+Tu:]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_m_to_g = torch.bmm(prob_c_m.unsqueeze(1), m_input_1hot).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_m_to_g
            selc_read_pz = cuda_(torch.zeros(u_input.size(0), 1, self.hidden_size))
        else:
            # copy from only user input, system response and previous z decoded
            score_c_pz = torch.tanh(self.Wcp_pz(pv_z_hidden))   # [B,Tz,H]
            score_c_pz = torch.bmm(score_c_pz, st.transpose(1, 2)).squeeze(2)      # [B,Tz]
            score_has_pv = torch.cat([score_g, score_c_u, score_c_m, score_c_pz],1) # [B, V+Tu+Tz]
            probs = F.softmax(score_has_pv, dim=1)
            prob_g, prob_c_u, prob_c_m = probs[:, :V], probs[:, V: V+Tu], probs[:, V+Tu: V+Tu+Tm]
            prob_c_pz = probs[:, V+Tu+Tm:]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_m_to_g = torch.bmm(prob_c_m.unsqueeze(1), m_input_1hot).squeeze(1)
            prob_c_pz_to_g = torch.bmm(prob_c_pz.unsqueeze(1), pv_z_prob).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_m_to_g + prob_c_pz_to_g  # [B,V]
            # compute selcive read from pv_z for the next step
            if self.enable_selc_read:
                selc_read_pz = get_selective_read(pv_z_idx, zt, pv_z_hidden, prob_c_pz)

        # compute selcive read from u and m for the next step
        if self.enable_selc_read:
            selc_read_u = get_selective_read(u_input, zt, u_hiddens, prob_c_u)
            selc_read_m = get_selective_read(m_input, zt, m_hiddens, prob_c_m)
        else:
            selc_read_u, selc_read_m, selc_read_pz = None, None, None

        if not return_gmb:
            return prob_out, last_hidden, gru_out, selc_read_u, selc_read_m, selc_read_pz, None
        else:
            if pv_z_prob is None:
                # copy from only user input and system response
                probs = gumbel_softmax(score_no_pv, temp)  # [B, V+Tu+Tm]
                prob_g, prob_c_u, prob_c_m = probs[:, :V], probs[:, V:V+Tu], probs[:, V+Tu:]
                prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
                prob_c_m_to_g = torch.bmm(prob_c_m.unsqueeze(1), m_input_1hot).squeeze(1)
                prob_out_gumbel = prob_g + prob_c_u_to_g + prob_c_m_to_g
            else:
                # copy from only user input, system response and previous z decoded
                probs = gumbel_softmax(score_has_pv, temp)
                prob_g, prob_c_u, prob_c_m = probs[:, :V], probs[:, V: V+Tu], probs[:, V+Tu: V+Tu+Tm]
                prob_c_pz = probs[:, V+Tu+Tm:]
                prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
                prob_c_m_to_g = torch.bmm(prob_c_m.unsqueeze(1), m_input_1hot).squeeze(1)
                prob_c_pz_to_g = torch.bmm(prob_c_pz.unsqueeze(1), pv_z_prob).squeeze(1)
                prob_out_gumbel = prob_g + prob_c_u_to_g + prob_c_m_to_g + prob_c_pz_to_g  # [B,V]
            if cfg.sample_type == 'ST_gumbel':
                z_sample = ST_gumbel_softmax_sample(prob_out_gumbel)
            elif cfg.sample_type == 'gumbel':
                z_sample = prob_out_gumbel
            else:
                z_sample = None
            return prob_out, last_hidden, gru_out, selc_read_u, selc_read_m, selc_read_pz, z_sample


# prior act decoder
class PriorDecoder_Pa(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, db_vec_size, slot_num,
                        dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size =2*hidden_size+ db_vec_size + slot_num
        self.dropout_rate = dropout_rate
        self.gru = nn.GRU(embed_size + hidden_size + db_vec_size + slot_num,
                                     hidden_size, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        # weights
        self.Wgen = nn.Linear(self.state_size, vocab_size) # generate mode
        self.attn = Attn(hidden_size)

    def forward(self, u_hiddens, emb_at, vec_input, last_hidden):
        """[summary]
        :param u_input: [B,Tu]
        :param u_input_1hot: [B,Tu,V]
        :param u_hiddens: [B,Tu,H]
        :param emb_at: [B,1,He]
        :param last_hidden: [1,B,H]
        """

        context = self.attn(last_hidden, u_hiddens)
        gru_input = torch.cat([emb_at, context, vec_input.unsqueeze(1)], dim=2)
        gru_input = self.dropout(gru_input)
        gru_out, last_hidden = self.gru(gru_input, last_hidden)  # gru_out: [B,1,H]
        st = torch.cat([gru_out, context, vec_input.unsqueeze(1)], dim=2)
        if cfg.dropout_st:
            st = self.dropout(st)
        score = self.Wgen(st).squeeze(1)     # [B,V]
        prob_out = F.softmax(score, dim=1)

        return prob_out, last_hidden, gru_out #, selc_read_z


# posterior act decoder
class PosteriorDecoder_Qa(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, db_vec_size,
        slot_num, dropout_rate, enable_selc_read=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = 2*hidden_size + db_vec_size + slot_num
        self.dropout_rate = dropout_rate
        self.enable_selc_read = enable_selc_read
        self.gru = nn.GRU(embed_size + hidden_size + db_vec_size + slot_num,  hidden_size, dropout=dropout_rate,
                                      batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        # weights
        if self.enable_selc_read:
            self.Win = nn.Linear(embed_size + hidden_size*3, embed_size)
        # self.Wst = nn.Linear(self.state_size, hidden_size)
        self.Wgen = nn.Linear(self.state_size, vocab_size) # generate mode
        self.Wcp_m = nn.Linear(hidden_size, self.state_size) # copy mode
        # self.Wcp_z = nn.Linear(hidden_size, self.state_size) # copy mode

        # self.attn_z = Attn(hidden_size)
        self.attn = Attn(hidden_size)

    def forward(self, u_hiddens, m_input, m_input_1hot, m_hiddens,
                        emb_at, vec_input, last_hidden, selc_read_m=None,
                        temp=0.1, return_gmb=False):
        V = m_input_1hot.size(2)

        if self.enable_selc_read:
            t_input = self.Win(torch.cat([emb_at, selc_read_u, selc_read_m], dim=2))
        else:
            t_input = emb_at   # [B,1,H]

        context = self.attn(last_hidden, torch.cat([u_hiddens, m_hiddens], dim=1))
        gru_input = torch.cat([t_input, context, vec_input.unsqueeze(1)], dim=2)
        gru_input = self.dropout(gru_input)

        gru_out, last_hidden = self.gru(gru_input, last_hidden)  # gru_out: [B,1,H]
        st = torch.cat([gru_out, context, vec_input.unsqueeze(1)], dim=2)
        if cfg.dropout_st:
            st = self.dropout(st)

        score_g = self.Wgen(st).squeeze(1)     # [B,V]
        score_c_m = torch.tanh(self.Wcp_m(m_hiddens))   # [B,Tu,H]
        score_c_m = torch.bmm(score_c_m, st.transpose(1, 2)).squeeze(2)      # [B,Tu]

        # copy from only user input and system response
        score = torch.cat([score_g, score_c_m],1) # [B, V+Tu+Tm]
        probs = F.softmax(score, dim=1)
        prob_g, prob_c_m = probs[:, :V], probs[:, V:]
        prob_c_m_to_g = torch.bmm(prob_c_m.unsqueeze(1), m_input_1hot).squeeze(1)
        prob_out = prob_g + prob_c_m_to_g

        # compute selcive read from u and m for the next step
        if self.enable_selc_read:
            selc_read_m = get_selective_read(m_input, zt, m_hiddens, prob_c_m)
        else:
            selc_read_m = None

        if not return_gmb:
            return prob_out, last_hidden, gru_out, selc_read_m, None
        else:
            # copy from only user input and system response
            probs = gumbel_softmax(score, temp)  # [B, V+Tu+Tm]
            prob_g, prob_c_m = probs[:, :V], probs[:, V:]
            prob_c_m_to_g = torch.bmm(prob_c_m.unsqueeze(1), m_input_1hot).squeeze(1)
            prob_out_gumbel = prob_g  + prob_c_m_to_g

            if cfg.sample_type == 'ST_gumbel':
                z_sample = ST_gumbel_softmax_sample(prob_out_gumbel)
            elif cfg.sample_type == 'gumbel':
                z_sample = prob_out_gumbel
            else:
                z_sample = None
            return prob_out, last_hidden, gru_out, selc_read_m, z_sample


class ResponseDecoder_Pm(nn.Module):
    def __init__(self, embed, embed_size, hidden_size, vocab_size, db_vec_size, dropout_rate,
                       enable_selc_read=False, model_act=False):
        super().__init__()
        self.embed = embed
        self.hidden_size = hidden_size
        self.model_act = model_act
        self.state_size = hidden_size*2 + db_vec_size
        self.enable_selc_read = enable_selc_read
        self.dropout_rate = dropout_rate
        self.gru = nn.GRU(embed_size + hidden_size  + db_vec_size, hidden_size,
                                      dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        # weights
        if self.enable_selc_read:
            input_size = embed_size + hidden_size*2
            if self.model_act:
                input_size += hidden_size
            self.Win = nn.Linear(input_size, hidden_size)
        # self.Wctx = nn.Linear(hidden_size*2, hidden_size)
        # self.Wst = nn.Linear(self.state_size, hidden_size)
        self.Wgen = nn.Linear(self.state_size, vocab_size) # generate mode
        self.Wcp_u = nn.Linear(hidden_size, self.state_size) # copy mode
        self.Wcp_z = nn.Linear(hidden_size, self.state_size) # copy mode
        if self.model_act:
            self.Wcp_a = nn.Linear(hidden_size, self.state_size) # copy mode

        self.attn = Attn(hidden_size)

    def forward(self, u_input, u_input_1hot, u_hiddens, z_input, pz_prob, z_hiddens, mt,
                        db_vec, last_hidden, a_input=None, pa_prob=None, a_hiddens=None,
                        selc_read_u=None, selc_read_z=None, selc_read_a=None):
        """[summary]
        :param u_input: [B,Tu]
        :param u_input_1hot: [B,Tu,V]
        :param u_hiddens: [B,Tu,H]
        :param z_input: [B, Tz]
        :param pz_prob: [B,Tz,V]
        :param z_hiddens: [B,Tz,H]
        :param mt: [B]
        :param selc_read_u: [B,1,H]
        :param selc_read_z: [B,1,H]
        :param db_vec: [B, db_vec_size]
        :param last_hidden: [1,B,H]
        """
        V = u_input_1hot.size(2)
        Tu = u_input.size(1)
        db_vec = db_vec.unsqueeze(1)

        if self.enable_selc_read:
            t_input = torch.cat([self.embed(mt), selc_read_u, selc_read_z], dim=2)
            if self.model_act:
                t_input = torch.cat([t_input, selc_read_a], dim=2)
            t_input = self.Win(t_input)
        else:
            t_input = self.embed(mt)   # [B,1,H]

        hiddens = [u_hiddens, z_hiddens]
        if self.model_act:
            hiddens.append(a_hiddens)
        context = self.attn(last_hidden, torch.cat(hiddens, dim=1))
        gru_input = torch.cat([t_input, context, db_vec], dim=2)
        if cfg.use_resp_dpout:
            gru_input = self.dropout(gru_input)

        gru_out, last_hidden = self.gru(gru_input, last_hidden)

        st = torch.cat([gru_out, context, db_vec], dim=2)
        if cfg.dropout_st and cfg.use_resp_dpout:
            st = self.dropout(st)
        score_g = self.Wgen(st).squeeze(1)     # [B,V]
        # score_g = self.Wgen(gru_out).squeeze(1)     # [B,V]

        # copy from user input and z decoded
        score_c_u = torch.tanh(self.Wcp_u(u_hiddens))   # [B,Tu,H]
        score_c_u = torch.bmm(score_c_u, st.transpose(1, 2)).squeeze(2)      # [B,Tu]
        score_c_z = torch.tanh(self.Wcp_z(z_hiddens))   # [B,Tz,H]
        score_c_z = torch.bmm(score_c_z, st.transpose(1, 2)).squeeze(2)      # [B,Tz]
        if not self.model_act:
            score = torch.cat([score_g, score_c_u, score_c_z],1) # [B, V+Tu+Tz]
            probs = F.softmax(score, dim=1)
            prob_g, prob_c_u, prob_c_z = probs[:, :V],  probs[:, V: V+Tu],  probs[:, V+Tu: ]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_z_to_g = torch.bmm(prob_c_z.unsqueeze(1), pz_prob).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_z_to_g  # [B,V]
        else:
            Tz = z_input.size(1)
            score_c_a = torch.tanh(self.Wcp_a(a_hiddens))   # [B,Tu,H]
            score_c_a = torch.bmm(score_c_a, st.transpose(1, 2)).squeeze(2)      # [B,Tu]
            score = torch.cat([score_g, score_c_u, score_c_z, score_c_a],1) # [B, V+Tu+Tz]
            probs = F.softmax(score, dim=1)
            prob_g, prob_c_u = probs[:, :V], probs[:, V: V+Tu]
            prob_c_z, prob_c_a =  probs[:, V+Tu : V+Tu+Tz], probs[:, V+Tu+Tz : ]
            prob_c_u_to_g = torch.bmm(prob_c_u.unsqueeze(1), u_input_1hot).squeeze(1)
            prob_c_z_to_g = torch.bmm(prob_c_z.unsqueeze(1), pz_prob).squeeze(1)
            prob_c_a_to_g = torch.bmm(prob_c_a.unsqueeze(1), pa_prob).squeeze(1)
            prob_out = prob_g + prob_c_u_to_g + prob_c_z_to_g + prob_c_a_to_g  # [B,V]

        # compute selcive read from u and z for the next step
        if self.enable_selc_read:
            selc_read_u = get_selective_read(u_input, mt, u_hiddens, prob_c_u)
            selc_read_z = get_selective_read(z_input, mt, z_hiddens, prob_c_z)
            if self.model_act:
                selc_read_a = get_selective_read(a_input, mt, a_hiddens, prob_c_a)
        else:
            selc_read_u, selc_read_z, selc_read_a = None, None, None

        return prob_out, last_hidden, gru_out, selc_read_u, selc_read_z, selc_read_a


# domain classifier
class DomainClassifier(nn.Module):
    def __init__(self, hidden_size, domain_num, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.W = nn.Linear(hidden_size, domain_num)

    def forward(self, u_hiddens):
        """[summary]
        :param u_hiddens: [B,Tu,H]
        """
        context = torch.mean(u_hiddens, dim=1)
        context = self.dropout(context)
        score = self.W(context)
        log_prob = F.log_softmax(score, dim=1)

        return log_prob