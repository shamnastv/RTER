import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import math

from model import RTERModel
from preprocess import preprocess
from util import to_torch_tesnsor

criterion = nn.CrossEntropyLoss()


def train(model, optimizer, train_data, device):
    model.train()

    feat, label, seq_len = train_data
    loss_accum = 0
    idx_train = np.random.permutation(len(feat))
    for i in idx_train:
        lb = torch.from_numpy(label[i]).to(device)
        pred = model(feat[i], seq_len[i])
        loss = criterion(pred, lb.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    return loss_accum


def validate(model, train_data, dev_data, test_data, device):
    model.eval()

    feat, label, seq_len = train_data
    idx_train = np.random.permutation(len(feat))
    train_correct = 0
    train_total = 0
    for i in idx_train:
        lb = torch.from_numpy(label[i]).to(device)
        pred = model(feat[i], seq_len[i]).max(1, keepdim=True)[1]
        train_correct += pred.eq(lb.view_as(pred)).sum().cpu().item()
        train_total += len(feat[i])

    acc_train = train_correct/train_total

    feat, label, seq_len = dev_data
    idx_dev = np.random.permutation(len(feat))
    dev_correct = 0
    dev_total = 0
    for i in idx_dev:
        lb = torch.from_numpy(label[i]).to(device)
        pred = model(feat[i], seq_len[i]).max(1, keepdim=True)[1]
        dev_correct += pred.eq(lb.view_as(pred)).sum().cpu().item()
        dev_total += len(feat[i])

    acc_dev = dev_correct/dev_total

    feat, label, seq_len = test_data
    idx_test = np.random.permutation(len(feat))
    test_correct = 0
    test_total = 0
    for i in idx_test:
        lb = torch.from_numpy(label[i]).to(device)
        pred = model(feat[i], seq_len[i]).max(1, keepdim=True)[1]
        test_correct += pred.eq(lb.view_as(pred)).sum().cpu().item()
        test_total += len(feat[i])

    acc_test = test_correct/test_total

    print("accuracy train: %f val: %f test: %f" % (acc_train, acc_dev, acc_test), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Pytorch for RTER')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension')
    parser.add_argument('--hops', type=int, default=1, help='hidden dimension')
    parser.add_argument('--wind_1', type=int, default=10, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-dataset', type=str, default='MELD', help='dataset')
    parser.add_argument('--dropout', type=float, default=0.5, help='learning rate (default: 0.01)')

    args = parser.parse_args()

    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    all_data_indexes, word_vectors, labels = preprocess(args.dataset)

    input_dim = word_vectors.shape[1]
    vocab_size = word_vectors.shape[0]
    num_classes = len(labels)

    word_embeddings = nn.Embedding(vocab_size, input_dim, padding_idx=1)
    word_embeddings.weight.data.copy_(torch.from_numpy(word_vectors))
    word_embeddings.weight.requires_grad = False

    train_data = to_torch_tesnsor(all_data_indexes['train'], device)
    dev_data = to_torch_tesnsor(all_data_indexes['dev'], device)
    test_data = to_torch_tesnsor(all_data_indexes['test'], device)

    model = RTERModel(args, input_dim, args.hidden_dim, num_classes, word_embeddings, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_data)
        validate(model, train_data, dev_data, test_data)


if __name__ == '__main__':
    main()