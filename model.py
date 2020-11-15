import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class UtteranceGRU(nn.Module):
    def __init__(self, in_dm, hidden_dm, num_layers, dropout):
        super(UtteranceGRU, self).__init__()
        self.gru = nn.GRU(input_size=in_dm, hidden_size=hidden_dm, bidirectional=True,
                          num_layers=num_layers, dropout=dropout)

    def forward(self):
        pass


class RTERModel(nn.Module):
    def __init__(self, args, input_dm, num_clasees, embeddings, device):
        super(RTERModel, self).__init__()
        self.num_classes = num_clasees
        self.hops = args.hops
        self.wind_1 = args.wind_1
        self.embeddings = embeddings
        self.device = device

        self.utt_gru = UtteranceGRU(input_dm, args.hidden_dim, args.num_layers, args.dropout)
        self.linear = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.dropout_utt = nn.Dropout(args.dropout)

        self.context_gru = nn.GRU(args.hidden_dim, args.hidden_dim, num_layers=1, bidirectional=True)
        self.dropout_context = nn.Dropout(0.3)

        self.attGRU = nn.ModuleList()
        for hop in range(self.hops):
            self.attGRU.append(AttGRU(d_model=100, bidirectional=True))

        self.classifier = nn.Linear(100, self.num_classes)

    def init_hidden(self, num_directs, num_layers, batch_size, d_model):
        return torch.zeros(num_directs * num_layers, batch_size, d_model)

    def forward(self, dialogue_ids):
        if len(dialogue_ids.size()) < 2:
            dialogue_ids.unsqueeze(0)
        dialogue = self.embeddings(dialogue_ids)

        dialogue_h = self.utt_gru(dialogue)
        max_pool = torch.max(dialogue_h, dim=1)[0]
        utterance_embd = self.linear(max_pool)
        utterance_embd = F.tanh(utterance_embd)
        utterance_embd = self.dropout_utt(utterance_embd)

        attn_weights = []
        masks = []
        batches = []
        s_out = []
        for i in range(utterance_embd.size()[0]):
            pad = max(self.wind_1 - i, 0)
            start = 0 if i < self.wind_1 + 1 else i - self.wind_1
            pad_tuple = [0, 0, 0, 0, pad, 0]
            m_pad = F.pad(utterance_embd[start:i], pad_tuple)
            batches.append(m_pad)
            mask = [1] * pad + [0] * (self.wind_1 - pad)
            masks.append(mask)

        batches = torch.cat(batches, dim=1)
        masks = torch.cat(masks, dim=1)

        q_mask = torch.ones(masks.size()[0], 1).long().to(self.device)

        a_mask = torch.matmul(q_mask.unsqueeze(2).float(), masks.unsqueeze(1).float()).eq(
            1).to(self.device)  # b_size x 1 x len_k

        mem_out = self.dropout_context(self.context_gru(batches)[0])
        mem_fwd, mem_bwd = mem_out.chunk(2, -1)
        mem_bank = (batches + mem_fwd + mem_bwd).transpose(0, 1).contiguous()

        query = utterance_embd[1:]
        eps_mem = query
        for hop in range(self.hops):
            attn_hid = torch.zeros(2, masks.size()[0], 100).to(self.device)
            attn_out, attn_weight = self.attGRU[hop](eps_mem, mem_bank, attn_hid, a_mask)
            attn_weights.append(attn_weight.squeeze(1))
            attn_out = self.dropout_mid(attn_out)
            attn_out1, attn_out2 = attn_out.chunk(2, -1)
            eps_mem = eps_mem + attn_out1 + attn_out2
        s_out.append(eps_mem)

        s_context = torch.cat(s_out, dim=0).squeeze(1)
        soutput = self.classifier(s_context)

        return soutput


class AttnGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttnGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

        self.initialize()

    def initialize(self):
        init.xavier_normal_(self.Wr.state_dict()['weight'])
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        init.xavier_normal_(self.W.state_dict()['weight'])
        init.xavier_normal_(self.U.state_dict()['weight'])

    def forward(self, c, hi_1, g):
        r_i = F.sigmoid(self.Wr(c) + self.Ur(hi_1))
        h_tilda = F.tanh(self.W(c) + r_i * self.U(hi_1))
        hi = g * h_tilda + (1 - g) * hi_1
        return hi


class AttGRU(nn.Module):
    def __init__(self, d_model, bidirectional):
        super(AttGRU, self).__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.gru = AttnGRUCell(100, 100)
        if self.bidirectional:
            self.gru_bwd = AttnGRUCell(100, 100)

    def forward(self, query, context, init_hidden, attn_mask=None):
        """
        :param query: batch x 1 x d_h
        :param context: batch x seq_len x d_h
        :param init_hidden: 1 x batch x d_h
        :param attn_mask: mask
        :return:
        """
        attn = torch.matmul(query, context.transpose(1, 2))
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -1e10)
        scores = F.softmax(attn, dim=-1)  # batch x 1 x seq_len

        # AttGRU summary
        hidden = init_hidden  # 1 x batch x d_h
        if self.bidirectional:
            hidden, hidden_bwd = init_hidden.chunk(2, 0)  # 2 x batch x d_h
        inp = context.transpose(0, 1).contiguous()  # seq_len x batch x d_h
        gates = scores.transpose(1, 2).transpose(0, 1).contiguous()  # seq_len x batch x 1
        seq_len = context.size()[1]
        for i in range(seq_len):
            hidden = self.gru(inp[i:i + 1], hidden, gates[i:i + 1])
            if self.bidirectional:
                hidden_bwd = self.gru_bwd(inp[seq_len - i - 1:seq_len - i], hidden_bwd,
                                          gates[seq_len - i - 1:seq_len - i])

        output = hidden.transpose(0, 1).contiguous()  # batch x 1 x d_h
        if self.bidirectional:
            output = torch.cat([hidden, hidden_bwd], dim=-1).transpose(0, 1).contiguous()  # batch x 1 x d_h*2

        return output, scores
