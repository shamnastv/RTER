import pickle
import re
import json
import unicodedata
from io import open
import numpy as np
import os

import fasttext

dir = 'data/'
data_splits = ['train', 'dev', 'test']
min_freq = 1
# max_len = 100
retrieve = False
save = True


def dump_data(dataset, data):
    with open(dataset, 'wb') as f:
        pickle.dump(data, f)


def read_data(dataset):
    f = open(dataset, 'rb')
    return pickle.load(f)


def utf_to_ascii(s):
    # s = ''.join(
    #     c for c in unicodedata.normalize('NFD', s.lower().strip())
    #     if unicodedata.category(c) != 'Mn'
    # ).strip()
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # return s
    string = s
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess(dataset):

    if retrieve and os.path.isfile(dataset):
        read_data(dataset)
    max_len = 0
    max_seq_l = 0
    word_to_id = {}
    word_list = ['<unk>', '<pad>']
    word_freq = {}
    labels = set()
    label_to_id = {}
    all_data = {}
    for ds in data_splits:
        filename = dir + dataset + '/' + dataset + '_' + ds + '.json'
        with open(filename, encoding='utf-8') as data_file:
            data = json.loads(data_file.read())

        dialogues = [[utf_to_ascii(utter['utterance']) for utter in dialog] for dialog in data]
        emotions = [[utter['emotion'] for utter in dialog] for dialog in data]

        all_data[ds] = (dialogues, emotions)

        for dia, emo in zip(dialogues, emotions):
            for d, e in zip(dia, emo):
                words = d.split()
                if len(words) > max_len:
                    max_len = len(words)

                for word in words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                labels.add(e.strip())

    labels = list(labels)
    labels.sort()
    print('max_len', max_len)

    for word in word_freq:
        if word_freq[word] >= min_freq:
            word_list.append(word)

    for i, word in enumerate(word_list):
        word_to_id[word] = i

    for i, label in enumerate(labels):
        label_to_id[label] = i

    all_data_indexes = {}

    print(len(word_freq))
    for split in data_splits:
        dialogues, emotions = all_data[split]
        dialogues_id = []
        emotions_id = []
        seq_lens = []
        for dia, emo in zip(dialogues, emotions):
            dia_id = []
            emo_id = []
            sq_len = []
            for d, e in zip(dia, emo):
                d_id = []
                # e_id = []
                words = d.split()
                for word in words:
                    if word in word_to_id:
                        d_id.append(word_to_id[word])
                    else:
                        d_id.append(word_to_id['<unk>'])

                seq_len = min(len(d_id), max_len)
                d_id += [word_to_id['<pad>']] * (max_len - seq_len)
                d_id = d_id[:max_len]

                if seq_len <= 0:
                    print(dia, e)
                    continue
                if seq_len > max_seq_l:
                    max_seq_l = seq_len

                emo_id.append(label_to_id[e])
                # e_id.append(label_to_id[e])
                dia_id.append(d_id)
                # emo_id.append(e_id)
                sq_len.append(seq_len)
            dialogues_id.append(dia_id)
            emotions_id.append(emo_id)
            seq_lens.append(sq_len)
        # print(dialogues_id[0][0])
        all_data_indexes[split] = (dialogues_id, emotions_id, seq_lens)

    print('max_seq_len', max_seq_l)
    # print(all_data_indexes)

    # print(len(word_list))
    word_vectors = get_vectors(word_list)
    # word_vectors = np.random.uniform(-0.1, 0.1, (len(word_list), 300))
    # if save:
    #     dump_data(dataset, (all_data_indexes, word_vectors, labels))

    return all_data_indexes, word_vectors, labels


def get_vectors(word_list):
    # print(len(word_list))
    # print(word_list)
    model = fasttext.load_model('model_new')
    word_vectors = [0, 1]
    for i in range(2, len(word_list)):
        word_vectors.append(model.get_word_vector(word_list[i]))
    model = None
    vec_dm = len(word_vectors[2])
    word_vectors[0] = np.random.uniform(-0.1, 0.1, vec_dm)
    word_vectors[1] = np.zeros(vec_dm, dtype=np.float32)
    # print(word_vectors[:5])
    word_vectors = np.array(word_vectors)

    sum_of_rows = word_vectors.sum(axis=1) + .0000001
    word_vectors = word_vectors / sum_of_rows[:, np.newaxis]

    return word_vectors


def main():
    preprocess('MELD')


if __name__ == '__main__':
    main()
