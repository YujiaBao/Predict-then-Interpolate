import os
import json
import random

from collections import defaultdict, Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchtext.vocab import Vocab, Vectors
from torch.distributions.categorical import Categorical


class MasterData():
    def __init__(self, file_path):
        self.text, self.attr, self.attr_names, self.all_text = self.load_json(file_path)

        random.seed(1)
        index = list(range(len(self.text)))
        random.shuffle(index)
        self.attr = self.attr[index]
        subtext = []
        for i in index:
            subtext.append(self.text[i])
        self.text = subtext

    def load_json(self, path):
        with open(path, 'r') as f:
            text = []
            attr_names = []
            attr = []
            all_text = []

            for line in f:
                example = json.loads(line)
                split_text = example['text'].lower().split()
                if len(split_text) > 500:
                    continue

                if len(attr_names) == 0:
                    for k in sorted(example.keys()):
                        if k != 'text':
                            attr_names.append(k)
                text.append(example['text'].lower())
                all_text.extend(split_text)

                cur_att = []
                for n in attr_names:
                    cur_att.append(example[n])
                attr.append(cur_att)

            attr = torch.tensor(attr)

        return text, attr, attr_names, all_text


class Pubmed(Dataset):
    def __init__(self, file_path, vocab=None):
        # load the entire master table
        self.data = MasterData(file_path)

        if vocab is not None:
            self.vocab = vocab
        else:
            path = './wiki.en.vec'
            if not os.path.exists(path):
                # Download the word vector and save it locally:
                print('Downloading word vectors')
                import urllib.request
                urllib.request.urlretrieve(
                    'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
                    path)

            # get word embeddings from fasttext
            vectors = Vectors('wiki.en.vec', cache='vector_cache')
            self.vocab = Vocab(Counter(self.data.all_text), vectors=vectors,
                          specials=['<pad>', '<unk>'], min_freq=5)

            # print word embedding statistics
            wv_size = self.vocab.vectors.size()
            print('Total num. of words: {}, word vector dimension: {}'.format(
                wv_size[0], wv_size[1]))

            num_oov = wv_size[0] - torch.nonzero(
                    torch.sum(torch.abs(self.vocab.vectors), dim=1), as_tuple=False).size()[0]
            print(('Num. of out-of-vocabulary words'
                   '(they are initialized to zeros): {}').format(num_oov))

        # get the idx of label and cor
        cor_name = 'breast cancer'
        self.label_idx = self.data.attr_names.index('penetrance')
        self.cor_idx = self.data.attr_names.index(cor_name)

        # define train / val / test split
        random.seed(1)
        train_list = list(range(int(len(self.data.text) * 0.5)))
        val_list = list(range(int(len(self.data.text) * 0.5),
                              int(len(self.data.text) * 0.7)))
        test_list = list(range(int(len(self.data.text) * 0.7),
                               int(len(self.data.text) * 1)))

        self.envs = [{'idx_list': []}, {'idx_list': []}]
        # define training environments based on the values of the spurious
        # attributes
        for idx in train_list:
            cor = self.data.attr[idx, self.cor_idx]
            if cor == 0:
                self.envs[0]['idx_list'].append(idx)
            else:
                self.envs[1]['idx_list'].append(idx)

        # define val and test environments
        self.envs.append({'idx_list': val_list})
        self.envs.append({'idx_list': test_list})

        # compute correlation between the given attribute cor and the target attribute
        # on the validation set for early stopping
        self.val_att_idx_dict = {
            cor_name: {'0_0': [], '0_1': [], '1_0': [], '1_1': []}
        }
        for i in val_list:
            k = '{}_{}'.format(self.data.attr[i, self.label_idx],
                               self.data.attr[i, self.cor_idx])
            self.val_att_idx_dict[cor_name][k].append(i)

        # compute correlation between each attribute and the target attribute
        # only for the test set
        self.test_att_idx_dict = {}
        for idx, att in enumerate(self.data.attr_names):
            if idx == self.label_idx:
                continue

            data_dict = {
                '0_0': [],
                '0_1': [],
                '1_0': [],
                '1_1': [],
            }

            # go through only the test examples
            for i in test_list:
                k = '{}_{}'.format(self.data.attr[i, self.label_idx],
                                   self.data.attr[i, idx])
                data_dict[k].append(i)

            # print data stats
            print('{:>20}'.format(att), end=' ')
            for k, v in data_dict.items():
                print(k, ' ', '{:>8}'.format(len(v)), end=', ')
            print()

            self.test_att_idx_dict[att] = data_dict

        self.length = len(self.data.attr)

    def __len__(self):
        return self.length

    def __getitem__(self, keys):
        idx = []
        for key in keys:
            env_id = int(key[1])  # this doesn't matter for Pubmed data
            idx.append(key[0])

        # idx = torch.tensor(idx).long()
        batch = {}
        batch['Y'] = self.data.attr[:, self.label_idx][idx]
        batch['C'] = self.data.attr[:, self.cor_idx][idx]
        batch['idx'] = torch.tensor(idx).long()

        # convert text into a dictionary of np arrays
        text_list = []
        for i in idx:
            text_list.append(self.data.text[i].split())

        text_len = np.array([len(text) for text in text_list])
        max_text_len = max(text_len)

        text = self.vocab.stoi['<pad>'] * np.ones(
            [len(text_list), max_text_len], dtype=np.int64)

        for i, t in enumerate(text_list):
            text[i, :len(t)] = [
                self.vocab.stoi[x] if x in self.vocab.stoi \
                else self.vocab.stoi['<unk>'] for x in t]

        batch['X'] = torch.tensor(text)
        batch['X_len'] = torch.tensor(text_len).long()

        return batch

    def get_all_y(self, env_id):
        all_y = self.data.attr[:, self.label_idx]
        return all_y[self.envs[env_id]['idx_list']].tolist()

    def get_all_c(self, env_id):
        all_c = self.data.attr[:, self.cor_idx]
        return all_c[self.envs[env_id]['idx_list']].tolist()

    def get_all_att(self, env_id):
        return self.data.attr

    def get_att_names(self, i):
        return self.data.attr_names[i]
