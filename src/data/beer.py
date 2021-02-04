import os
import json
import random
from collections import defaultdict, Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchtext.vocab import Vocab, Vectors
from torch.distributions.categorical import Categorical


class BeerReview(Dataset):
    def __init__(self, file_path, val, aspect, vocab=None):
        # load the beer review data
        print('Load beer data for aspect {}'.format(aspect))

        # iterate through the envs
        self.envs = []
        all_words = []
        self.length = 0

        for i in range(4):
            if i == 2:
                # choose validation env
                if val == 'in_domain':
                    data, words = BeerReview.load_json(os.path.join(
                        file_path, 'art_aspect_{}_env_1_val.json'.format(aspect)))
                else:
                    data, words = BeerReview.load_json(os.path.join(
                        file_path, 'art_aspect_{}_env_2_val.json'.format(aspect)))
            elif i == 3:  # test env
                data, words = BeerReview.load_json(os.path.join(
                    file_path, 'art_aspect_{}_env_2.json'.format(aspect)))
            else:
                data, words = BeerReview.load_json(os.path.join(
                    file_path, 'art_aspect_{}_env_{}.json'.format(aspect, i)))

            self.envs.append(data)

            all_words.extend(words)
            self.length += len(data['y'])

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
            self.vocab = Vocab(Counter(all_words), vectors=vectors,
                          specials=['<pad>', '<unk>', '<art_negative>',
                                    '<art_positive>'], min_freq=5)

            # randomly initalize embeedings for the spurious tokens
            self.vocab.vectors[self.vocab.stoi['<art_negative>']] = torch.rand(300)
            self.vocab.vectors[self.vocab.stoi['<art_positive>']] = torch.rand(300)

            # print word embedding statistics
            wv_size = self.vocab.vectors.size()
            print('Total num. of words: {}, word vector dimension: {}'.format(
                wv_size[0], wv_size[1]))

            num_oov = wv_size[0] - torch.nonzero(
                    torch.sum(torch.abs(self.vocab.vectors), dim=1), as_tuple=False).size()[0]
            print(('Num. of out-of-vocabulary words'
                   '(they are initialized to zeros): {}').format(num_oov))

        # not evaluating worst-case performance for beer
        self.val_att_idx_dict = None
        self.test_att_idx_dict = None

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            data = {'y': [], 'c': [], 'x': []}

            all_text = []

            for line in f:
                example = json.loads(line)
                data['y'].append(example['y'])
                data['x'].append(example['text'])
                data['c'].append(example['c'])
                all_text.extend(example['text'].split())

            data['y'] = torch.tensor(data['y'])
            data['c'] = torch.tensor(data['c'])
            data['idx_list'] = list(range(len(data['y'])))

        return data, all_text

    def __len__(self):
        return self.length

    def __getitem__(self, keys):
        idx = []
        for key in keys:
            env_id = int(key[1])
            idx.append(key[0])

        # get labels
        batch = {}
        batch['Y'] = self.envs[env_id]['y'][idx]
        batch['C'] = self.envs[env_id]['c'][idx]
        batch['idx'] = torch.tensor(idx).long()

        # convert text into a dictionary of np arrays
        text_list = []
        for i in idx:
            text_list.append(self.envs[env_id]['x'][i].split())

        text_len = np.array([len(text) for text in text_list])
        max_text_len = max(text_len)

        # initialize the big numpy array by <pad>
        text = self.vocab.stoi['<pad>'] * np.ones(
            [len(text_list), max_text_len], dtype=np.int64)

        # convert each token to its corresponding id
        for i, t in enumerate(text_list):
            text[i, :len(t)] = [
                self.vocab.stoi[x] if x in self.vocab.stoi \
                else self.vocab.stoi['<unk>'] for x in t]

        batch['X'] = torch.tensor(text)
        batch['X_len'] = torch.tensor(text_len).long()

        return batch

    def get_all_y(self, env_id):
        return self.envs[env_id]['y'].tolist()

    def get_all_c(self, env_id):
        return self.envs[env_id]['c'].tolist()
