import os
import json
import random

from collections import defaultdict, Counter

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
from torch.distributions.categorical import Categorical


class Celeba(Dataset):
    def __init__(self, file_path):
        # load the entire celeba data
        self.train_data = datasets.CelebA(file_path, split='train',
                                          target_type='attr', transform=None,
                                          target_transform=None, download=True)

        self.val_data = datasets.CelebA(file_path, split='valid',
                                        target_type='attr', transform=None,
                                        target_transform=None, download=True)

        self.test_data = datasets.CelebA(file_path, split='test',
                                         target_type='attr', transform=None,
                                         target_transform=None, download=True)

        # get the idx of label and cor
        cor_name = 'Male'
        self.label_idx = self.train_data.attr_names.index('Blond_Hair')
        self.cor_idx = self.train_data.attr_names.index(cor_name)

        # def train environments
        self.envs = [
            {
                'idx_list': [],
                'data': self.train_data
            },
            {
                'idx_list': [],
                'data': self.train_data
            },
        ]

        # obtain training environments based on the provided attribute
        for idx in range(len(self.train_data.attr)):
            if self.train_data.attr[idx, self.cor_idx] == 0:
                self.envs[0]['idx_list'].append(idx)
            else:
                self.envs[1]['idx_list'].append(idx)

        # def validation environment
        self.envs.append({
            'idx_list': list(range(len(self.val_data.attr))),
            'data': self.val_data,
        })

        # def test environment
        self.envs.append({
            'idx_list': list(range(len(self.test_data.attr))),
            'data': self.test_data,
        })

        # for val environment, compute the mask for the given attribute
        self.val_att_idx_dict = {
            cor_name: { '0_0': [], '0_1': [], '1_0': [], '1_1': []}
        }
        for i in range(len(self.val_data.attr)):
            k = '{}_{}'.format(self.val_data.attr[i, self.label_idx],
                               self.val_data.attr[i, self.cor_idx])
            self.val_att_idx_dict[cor_name][k].append(i)

        # compute correlation between each attribute and the target attribute
        # only for the test set
        self.test_att_idx_dict = {}
        for idx, att in enumerate(self.test_data.attr_names):
            if idx == self.label_idx:
                continue

            data_dict = {
                '0_0': [],
                '0_1': [],
                '1_0': [],
                '1_1': [],
            }

            # go through only the att label
            for i, attrs in enumerate(self.test_data.attr):
                k = '{}_{}'.format(attrs[self.label_idx], attrs[idx])
                data_dict[k].append(i)

            # print data stats
            print('{:>20}'.format(att), end=' ')
            for k, v in data_dict.items():
                print(k, ' ', '{:>8}'.format(len(v)), end=', ')
            print()

            self.test_att_idx_dict[att] = data_dict

        self.length = len(self.train_data.attr) + len(self.val_data.attr) + len(self.test_data.attr)


    def __len__(self):
        return self.length

    def __getitem__(self, keys):
        idx = []
        for key in keys:
            env_id = int(key[1])  # this doesn't matter for Pubmed data
            idx.append(key[0])

        batch = {}
        batch['Y'] = self.envs[env_id]['data'].attr[:, self.label_idx][idx]
        batch['C'] = self.envs[env_id]['data'].attr[:, self.cor_idx][idx]
        batch['idx'] = torch.tensor(idx).long()

        # convert text into a dictionary of np arrays
        img2tensor = transforms.ToTensor()
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        x = []
        for i in idx:
            img = img2tensor(self.envs[env_id]['data'][i][0])
            img = transform(img)
            x.append(img)

        batch['X'] = torch.stack(x)

        return batch

    def get_all_y(self, env_id):
        return self.envs[env_id]['data'].attr[self.envs[env_id]['idx_list'], self.label_idx].tolist()

    def get_all_c(self, env_id):
        return self.envs[env_id]['data'].attr[self.envs[env_id]['idx_list'],
                                              self.cor_idx].tolist()

    def get_all_att(self, env_id):
        return self.envs[env_id]['data'].attr

    def get_att_names(self, i):
        return self.envs[0]['data'].attr_names[i]
