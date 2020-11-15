import pickle
import re
import json
import unicodedata
from io import open
import numpy as np

import fasttext

dir = 'data/'
data_splits = ['train', 'dev', 'test']
min_freq = 3
max_len = 30


def dump_data(dataset, data_split, data):
    with open(dataset + data_split, 'wb') as f:
        pickle.dump(data, f)


def read_data(dataset, data_split):
    f = open(dataset + data_split, 'rb')
    return pickle.load(f)


def utf_to_ascii(s):
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    ).strip()
    s = re.sub(r"([!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s


def preprocess(dataset):
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
                for word in words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                labels.add(e.strip())

    for word in word_freq:
        if word_freq[word] >= min_freq:
            word_list.append(word)

    for i, word in enumerate(word_list):
        word_to_id[word] = i

    for i, label in enumerate(labels):
        label_to_id[label] = i

    all_data_indexes = {}

    for split in data_splits:
        dialogues, emotions = all_data[split]
        dialogues_id = []
        emotions_id = []
        for dia, emo in zip(dialogues, emotions):
            dia_id = []
            emo_id = []
            for d, e in zip(dia, emo):
                d_id = []
                e_id = []
                words = d.split()
                for word in words:
                    if word in word_to_id:
                        d_id.append(word_to_id[word])
                    else:
                        d_id.append(word_to_id['<unk>'])

                if len(d_id) > max_len:
                    d_id = d_id[:max_len]
                else:
                    d_id += [word_to_id['<pad>']] * (max_len - len(d_id))

                e_id.append(label_to_id[e])
                dia_id.append(d_id)
                emo_id.append(e_id)
            dialogues_id.append(dia_id)
            emotions_id.append(emo_id)
        all_data_indexes[split] = (dialogues_id, emotions_id)

    # print(all_data_indexes)

    print(len(word_list))
    # word_vectors = get_vectors(word_list)
    word_vectors = None

    return all_data_indexes, word_vectors, labels


def get_vectors(word_list):
    model = fasttext.load_model('model')
    word_vectors = [0, 1]
    for i in range(2, len(word_list)):
        word_vectors.append(model.get_word_vector(word_list[i]))
    model = None
    vec_dm = len(word_list[2])
    word_list[0] = np.random.uniform(-0.01, 0.01, vec_dm)
    word_vectors[2] = np.zeros(vec_dm)
    word_vectors = np.array(word_vectors)
    return word_vectors


def main():
    preprocess('MELD')


if __name__ == '__main__':
    main()
