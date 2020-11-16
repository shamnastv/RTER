import torch
import numpy as np


def to_torch_tensor(data):
    feat, label, seq_len = data

    for i in range(len(feat)):
        # print(label[i])
        if 0 in seq_len[i]:
            print(i)
        feat[i] = torch.LongTensor(feat[i])
        label[i] = np.array(label[i])
        seq_len[i] = np.array(seq_len[i])

    return feat, label, seq_len
