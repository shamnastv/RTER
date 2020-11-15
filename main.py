import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import math

from preprocess import preprocess


def main():
    parser = argparse.ArgumentParser(description='Pytorch for RTER')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('-dataset', type=str, default='Friends', help='dataset')

    args = parser.parse_args()

    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    all_data_indexes, word_vectors, labels = preprocess(args.dataset)

    word_embeddings = nn.Embedding(word_vectors.shape[0], word_vectors.shape[1], padding_idx=1)
    word_embeddings.weight.data.copy_(torch.from_numpy(word_vectors))
    word_embeddings.weight.requires_grad = False


