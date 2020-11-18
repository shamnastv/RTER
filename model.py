import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class UtteranceGRU(nn.Module):
    def __init__(self, in_dm, hidden_dim, num_layers, dropout, device):
        super(UtteranceGRU, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_size=in_dm, hidden_size=hidden_dim, bidirectional=True,
                          num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dialogue, seq_lens):

        # uttr_lengths_sorted = np.sort(seq_lens)[::-1].copy()
        # index_uttr_lengths_sorted = np.argsort(-seq_lens)
        # index_uttr_lengths_unsort = np.argsort(index_uttr_lengths_sorted)
        # index_uttr_lengths_sorted = torch.from_numpy(index_uttr_lengths_sorted).to(self.device)
        # index_uttr_lengths_unsort = torch.from_numpy(index_uttr_lengths_unsort).to(self.device)
        #
        # dialogue_sorted = dialogue.transpose(0, 1).index_select(1, index_uttr_lengths_sorted)
        #
        # dialogue_packed = pack_padded_sequence(dialogue_sorted, uttr_lengths_sorted)
        # dialogue_embd = self.gru(dialogue_packed)[0]
        # dialogue_embd = pad_packed_sequence(dialogue_embd, total_length=dialogue.shape[1])[0]
        #
        # dialogue_embd = dialogue_embd.index_select(1, index_uttr_lengths_unsort).transpose(0, 1)

        uttr_lengths = torch.from_numpy(seq_lens).to(self.device)
        uttr_lengths, sorted_indices = torch.sort(uttr_lengths, descending=True)
        sorted_indices = sorted_indices.to(self.device)
        _, unsorted_indices = torch.sort(sorted_indices, descending=False)

        dialogue_sorted = dialogue.index_select(0, sorted_indices)

        dialogue_packed = pack_padded_sequence(dialogue_sorted, uttr_lengths, batch_first=True)
        dialogue_embd = self.gru(dialogue_packed)[0]
        dialogue_embd = pad_packed_sequence(dialogue_embd, batch_first=True, total_length=dialogue.shape[1])[0]
        dialogue_embd = dialogue_embd.index_select(0, unsorted_indices)

        utterance_embd = torch.max(dialogue_embd, dim=1)[0]
        utterance_embd = self.linear(utterance_embd)
        utterance_embd = torch.tanh(utterance_embd)
        utterance_embd = self.dropout(utterance_embd).unsqueeze(1)

        return utterance_embd


class RTERModel(nn.Module):
    def __init__(self, args, input_dm, hidden_dim, num_clasees, embeddings, device):
        super(RTERModel, self).__init__()
        self.num_classes = num_clasees
        self.hops = args.hops
        self.max_window_size = args.max_window_size
        self.embeddings = embeddings
        self.device = device
        self.hidden_dim = hidden_dim

        self.utt_gru = UtteranceGRU(input_dm, hidden_dim, args.num_layers, args.dropout, device)

        self.context_gru = nn.GRU(args.hidden_dim, args.hidden_dim, num_layers=1, bidirectional=True)
        self.dropout_context = nn.Dropout(0.3)

        self.attGRU = nn.ModuleList()
        for hop in range(self.hops):
            self.attGRU.append(AttGRU(hidden_dim=hidden_dim, bidirectional=True))

        self.classifier = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, dialogue_ids, seq_lens):
        if len(dialogue_ids.size()) < 2:
            dialogue_ids.unsqueeze(0)
        dialogue_ids = dialogue_ids.to(self.device)
        dialogue = self.embeddings(dialogue_ids)

        utterance_embd = self.utt_gru(dialogue, seq_lens)

        s_out = [utterance_embd[:1]]

        masks = []
        batches = []
        window_size = min(self.max_window_size, utterance_embd.size()[0] - 1)
        for i in range(1, utterance_embd.size()[0]):
            padding = max(window_size - i, 0)
            start = 0 if i < window_size + 1 else i - window_size
            pad_tuple = [0, 0, 0, 0, padding, 0]
            memory_padded = F.pad(utterance_embd[start:i], pad_tuple)
            batches.append(memory_padded)
            mask = [1] * padding + [0] * (window_size - padding)
            masks.append(mask)

        if len(batches) > 0:
            batches = torch.cat(batches, dim=1)
            masks = torch.tensor(masks).long().to(self.device)

            q_mask = torch.ones(masks.size()[0], 1).long().to(self.device)

            a_mask = torch.matmul(q_mask.unsqueeze(2).float(), masks.unsqueeze(1).float()).eq(
                1).to(self.device)  # b_size x 1 x len_k

            mem_out = self.dropout_context(self.context_gru(batches)[0])
            mem_fwd, mem_bwd = mem_out.chunk(2, -1)
            mem_bank = (batches + mem_fwd + mem_bwd).transpose(0, 1).contiguous()

            query = utterance_embd[1:]
            for hop in range(self.hops):
                attn_hid = torch.zeros(2, masks.size()[0], self.hidden_dim).to(self.device)
                attn_out = self.attGRU[hop](query, mem_bank, attn_hid, a_mask)
                attn_out = self.dropout_context(attn_out)
                attn_out1, attn_out2 = attn_out.chunk(2, -1)
                query = query + attn_out1 + attn_out2
            s_out.append(query)

        s_context = torch.cat(s_out, dim=0).squeeze(1)
        soutput = self.classifier(s_context)

        return soutput


class AttnGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttnGRUCell, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.Wr = nn.Linear(input_dim, hidden_dim)
        self.Ur = nn.Linear(hidden_dim, hidden_dim)
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)

        self.initialize()

    def initialize(self):
        init.xavier_normal_(self.Wr.state_dict()['weight'])
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        init.xavier_normal_(self.W.state_dict()['weight'])
        init.xavier_normal_(self.U.state_dict()['weight'])

    def forward(self, c, hi_1, g):
        r_i = torch.sigmoid(self.Wr(c) + self.Ur(hi_1))
        h_tilda = torch.tanh(self.W(c) + r_i * self.U(hi_1))
        hi = g * h_tilda + (1 - g) * hi_1
        return hi


class AttGRU(nn.Module):
    def __init__(self, hidden_dim, bidirectional):
        super(AttGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.gru = AttnGRUCell(hidden_dim, hidden_dim)
        if self.bidirectional:
            self.gru_bwd = AttnGRUCell(hidden_dim, hidden_dim)

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

        return output
