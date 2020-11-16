import torch


def to_torch_tesnsor(data, device):
    feat, label, seq_len = data

    for i in range(len(feat)):
        # print(label[i])
        feat[i] = torch.LongTensor(feat[i]).to(device)
        label[i] = torch.LongTensor(label[i]).to(device)
        seq_len[i] = torch.LongTensor(seq_len[i]).to(device)

    return feat, label, seq_len
