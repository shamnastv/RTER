import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attention import Attention


class UtteranceGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, device):
        super(UtteranceGRU, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True,
                          num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim * 2)

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

        att_w = F.softmax(self.attention(dialogue_embd))
        utterance_embd = torch.matmul(att_w.transpose(1, 2), dialogue_embd).squeeze(1)

        # utterance_embd = torch.max(dialogue_embd, dim=1)[0]
        utterance_embd = self.linear(utterance_embd)
        # utterance_embd = torch.tanh(utterance_embd)
        utterance_embd = F.leaky_relu(utterance_embd)
        utterance_embd = self.dropout(utterance_embd)

        return utterance_embd


class AttnGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttnGRUCell, self).__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.W_ir = nn.Linear(input_dim, hidden_dim)
        self.W_hr = nn.Linear(hidden_dim, hidden_dim)
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_hn = nn.Linear(hidden_dim, hidden_dim)
        self.W_iz = nn.Linear(input_dim, hidden_dim)
        self.W_hz = nn.Linear(hidden_dim, hidden_dim)

        # self.initialize()

    def initialize(self):
        init.xavier_normal_(self.W_ir.state_dict()['weight'])
        init.xavier_normal_(self.W_hr.state_dict()['weight'])
        init.xavier_normal_(self.W_in.state_dict()['weight'])
        init.xavier_normal_(self.W_hn.state_dict()['weight'])

    def forward(self, c, ht_1, g):
        r_t = torch.sigmoid(self.W_ir(c) + self.W_hr(ht_1))
        z_t = torch.sigmoid(self.W_iz(c) + self.W_hz(ht_1))
        g = torch.sigmoid(g + z_t)
        n_t = torch.tanh(self.W_in(c) + r_t * self.W_hn(ht_1))
        h_t = g * n_t + (1 - g) * ht_1
        return h_t


class AttGRU(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(AttGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_fwd = AttnGRUCell(hidden_dim, hidden_dim)
        self.gru_bwd = AttnGRUCell(hidden_dim, hidden_dim)

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, hidden, attn_mask):
        """
        :param query: batch x 1 x d_h
        :param memory: batch x seq_len x d_h
        :param hidden: 1 x batch x d_h
        :param attn_mask: mask
        :return:
        """
        memory_t = memory.transpose(0, 1).contiguous()  # seq_len x batch x d_h
        seq_len = memory_t.size(0)
        hidden_fwd, hidden_bwd = hidden.chunk(2, 0)  # 2 x batch x d_h

        attention_weights = torch.matmul(self.W(query), self.U(memory).transpose(1, 2))
        attention_weights.data.masked_fill_(attn_mask, -np.inf)
        attention_weights = F.softmax(attention_weights, dim=-1)  # batch x 1 x seq_len
        gates = attention_weights.transpose(1, 2).transpose(0, 1).contiguous()  # seq_len x batch x 1

        for i in range(seq_len):
            hidden_fwd = self.gru_fwd(memory_t[i:i + 1], hidden_fwd, gates[i:i + 1])
            hidden_bwd = self.gru_bwd(memory_t[seq_len - i - 1:seq_len - i],
                                      hidden_bwd, gates[seq_len - i - 1:seq_len - i])

        output = torch.cat([hidden_fwd, hidden_bwd], dim=-1).transpose(0, 1).contiguous()  # batch x 1 x d_h*2

        output = self.dropout(output)
        return output.chunk(2, -1)


class RTERModel(nn.Module):
    def __init__(self, args, input_dm, hidden_dim, num_clasees, word_embeddings, device):
        super(RTERModel, self).__init__()
        self.num_classes = num_clasees
        self.hops = args.hops
        self.max_window_size = args.max_window_size
        self.word_embeddings = word_embeddings
        self.device = device
        self.hidden_dim = hidden_dim

        self.utt_gru = UtteranceGRU(input_dm, hidden_dim, args.num_layers, args.dropout, device)

        self.fusion_gru = nn.GRU(input_size=args.hidden_dim, hidden_size=args.hidden_dim, bidirectional=True)
        self.fusion_dropout = nn.Dropout(args.dropout)

        self.attGRU = nn.ModuleList()
        for hop in range(self.hops):
            self.attGRU.append(AttGRU(hidden_dim=hidden_dim, dropout=args.dropout))

        self.classifier = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, dialogue_ids, seq_lens):
        if len(dialogue_ids.size()) < 2:
            dialogue_ids.unsqueeze(0)
        dialogue_ids = dialogue_ids.to(self.device)
        dialogue = self.word_embeddings(dialogue_ids)

        utterance_embd = self.utt_gru(dialogue, seq_lens)

        masks = []
        batches = []
        window_size = min(self.max_window_size, utterance_embd.size()[0] - 1)

        utterance_embd = utterance_embd.unsqueeze(1)
        uttr_embd_with_memory = [utterance_embd[:1]]

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
            fusion_ouput = self.fusion_gru(batches)[0]
            fusion_ouput = self.fusion_dropout(fusion_ouput)
            fusion_output_fwd, fusion_output_bwd = fusion_ouput.chunk(2, -1)
            memory_bank = (batches + fusion_output_fwd + fusion_output_bwd).transpose(0, 1).contiguous()

            masks = torch.tensor(masks).long().to(self.device)
            attention_mask = masks.unsqueeze(1).eq(1)

            # masks = torch.tensor(masks).long().to(self.device)
            # q_mask = torch.ones(masks.size()[0], 1).long().to(self.device)
            # b_mask = torch.matmul(q_mask.unsqueeze(2).float(), masks.unsqueeze(1).float()).eq(
            #     1).to(self.device)  # b_size x 1 x len_k
            # if not torch.all(torch.eq(a_mask, b_mask)):
            #     print('not equal')

            query = utterance_embd[1:]
            for hop in range(self.hops):
                attn_hidden = torch.zeros(2, masks.size()[0], self.hidden_dim).to(self.device)
                attn_output_fwd, attn_output_bwd = self.attGRU[hop](query, memory_bank, attn_hidden, attention_mask)
                query = query + attn_output_fwd + attn_output_bwd
            uttr_embd_with_memory.append(query)

        uttr_embd_with_memory = torch.cat(uttr_embd_with_memory, dim=0).squeeze(1)
        uttr_classes = self.classifier(uttr_embd_with_memory)

        return uttr_classes
