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
from util import to_torch_tensor

criterion = nn.CrossEntropyLoss()
start_time = time.time()


def print_distr(x):
    dst = [0] * (max(x) + 1)
    for i in x:
        dst[i] += 1
    print(dst)


def print_distr_y(y, label, device):
    print('Class distributions')
    x = []
    for e in y:
        x += e
    freq = [0] * len(set(x))
    for i in x:
        freq[i] += 1

    print(label)
    print(freq)
    m = 1
    for i in freq:
        if i < m:
            m = i

    weights = [m / i if i != 0 else 1 for i in freq]
    global criterion
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))


def train(epoch, model, optimizer, train_data, device):
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

    print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))

    return loss_accum


def validate(epoch, model, train_data, dev_data, test_data, label_list, device, print_f1):
    model.eval()

    num_labels = len(label_list)
    print(label_list)

    tp = np.zeros(num_labels)
    tp_fp = np.zeros(num_labels)
    tp_fn = np.zeros(num_labels)

    feat, label, seq_len = train_data
    train_correct = 0
    train_total = 0
    for i in range(len(feat)):
        true_label = torch.from_numpy(label[i]).to(device)
        with torch.no_grad():
            pred = model(feat[i], seq_len[i]).max(1, keepdim=True)[1]
        train_correct += pred.eq(true_label.view_as(pred)).sum().cpu().item()
        train_total += len(feat[i])

        if print_f1:
            true_label = true_label.reshape(-1)
            pred = pred.reshape(-1)
            for j in range(len(true_label)):
                idx = true_label[j].item()
                tp_fn[idx] += 1
                if true_label[j] == pred[j]:
                    tp[idx] += 1
                tp_fp[pred[j]] += 1

    if print_f1:
        recall = [np.round(itp / itp_fn * 100, 2) if itp_fn > 0 else 0 for itp, itp_fn in zip(tp, tp_fn)]
        precision = [np.round(itp / itp_fp * 100, 2) if itp_fp > 0 else 0 for itp, itp_fp in zip(tp, tp_fp)]
        f1 = [np.round(2 * r * p / (r + p), 2) if r + p > 0 else 0 for r, p in zip(recall, precision)]
        print('train f1 : ', f1)

    acc_train = train_correct/train_total

    tp = np.zeros(num_labels)
    tp_fp = np.zeros(num_labels)
    tp_fn = np.zeros(num_labels)

    feat, label, seq_len = dev_data
    dev_correct = 0
    dev_total = 0
    for i in range(len(feat)):
        true_label = torch.from_numpy(label[i]).to(device)
        with torch.no_grad():
            pred = model(feat[i], seq_len[i]).max(1, keepdim=True)[1]
        dev_correct += pred.eq(true_label.view_as(pred)).sum().cpu().item()
        dev_total += len(feat[i])

        if print_f1:
            true_label = true_label.reshape(-1)
            pred = pred.reshape(-1)
            for j in range(len(true_label)):
                idx = true_label[j].item()
                tp_fn[idx] += 1
                if true_label[j] == pred[j]:
                    tp[idx] += 1
                tp_fp[pred[j]] += 1
    if print_f1:
        recall = [np.round(itp / itp_fn * 100, 2) if itp_fn > 0 else 0 for itp, itp_fn in zip(tp, tp_fn)]
        precision = [np.round(itp / itp_fp * 100, 2) if itp_fp > 0 else 0 for itp, itp_fp in zip(tp, tp_fp)]
        f1 = [np.round(2 * r * p / (r + p), 2) if r + p > 0 else 0 for r, p in zip(recall, precision)]
        print('dev f1 : ', f1)

    acc_dev = dev_correct/dev_total

    tp = np.zeros(num_labels)
    tp_fp = np.zeros(num_labels)
    tp_fn = np.zeros(num_labels)

    feat, label, seq_len = test_data
    test_correct = 0
    test_total = 0
    test_labels = []
    test_pred = []
    for i in range(len(feat)):
        true_label = torch.from_numpy(label[i]).to(device)
        with torch.no_grad():
            pred = model(feat[i], seq_len[i]).max(1, keepdim=True)[1]
        test_correct += pred.eq(true_label.view_as(pred)).sum().cpu().item()
        test_total += len(feat[i])

        if epoch % 10 == 0:
            test_labels.append(true_label.reshape(-1))
            test_pred.append(pred.reshape(-1))

        if print_f1:
            true_label = true_label.reshape(-1)
            pred = pred.reshape(-1)
            for j in range(len(true_label)):
                idx = true_label[j].item()
                tp_fn[idx] += 1
                if true_label[j] == pred[j]:
                    tp[idx] += 1
                tp_fp[pred[j]] += 1

    if print_f1:
        recall = [np.round(itp / itp_fn * 100, 2) if itp_fn > 0 else 0 for itp, itp_fn in zip(tp, tp_fn)]
        precision = [np.round(itp / itp_fp * 100, 2) if itp_fp > 0 else 0 for itp, itp_fp in zip(tp, tp_fp)]
        f1 = [np.round(2 * r * p / (r + p), 2) if r + p > 0 else 0 for r, p in zip(recall, precision)]
        print('test f1 : ', f1)

    acc_test = test_correct/test_total

    print("accuracy train: %f val: %f test: %f" % (acc_train, acc_dev, acc_test), flush=True)

    if epoch % 10 == 0:
        test_labels = torch.cat(test_labels).cpu().numpy()
        test_pred = torch.cat(test_pred).cpu().numpy()
        # for i in range(len(test_labels)):
        #     print(test_labels[i], test_pred[i])
        print_distr(test_labels)
        print_distr(test_pred)


def main():
    parser = argparse.ArgumentParser(description='Pytorch for RTER')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension')
    parser.add_argument('--hops', type=int, default=1, help='hidden dimension')
    parser.add_argument('--max_window_size', type=int, default=40, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, default='MELD', help='dataset')
    parser.add_argument('--dropout', type=float, default=0.3, help='learning rate (default: 0.01)')
    parser.add_argument('--print_f1', action="store_true", help='print f1 score')

    args = parser.parse_args()

    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    all_data_indexes, word_vectors, label_list = preprocess(args.dataset)
    print_distr_y(all_data_indexes['train'][1], label_list, device)

    print(label_list)

    input_dim = word_vectors.shape[1]
    vocab_size = word_vectors.shape[0]
    num_classes = len(label_list)

    word_embeddings = nn.Embedding(vocab_size, input_dim, padding_idx=1)
    word_embeddings.weight.data.copy_(torch.from_numpy(word_vectors))
    word_embeddings.weight.requires_grad = False

    train_data = to_torch_tensor(all_data_indexes['train'])
    dev_data = to_torch_tensor(all_data_indexes['dev'])
    test_data = to_torch_tensor(all_data_indexes['test'])

    model = RTERModel(args, input_dim, args.hidden_dim, num_classes, word_embeddings, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, train_data, device)
        validate(epoch, model, train_data, dev_data, test_data, label_list, device, args.print_f1)
        print('')


if __name__ == '__main__':
    main()