import random
from collections import defaultdict

import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical


class ColoredMNIST(Dataset):
    def __init__(self, file_path, is_train, val, env0=None, env1=None):
        print('Load data')
        mnist = datasets.MNIST(file_path, train=is_train, download=True)

        print('Create per class data dictionary')

        self.data = defaultdict(list)
        label_dict = dict(zip(list(range(10)), list(range(10))))

        for x, y in zip(mnist.data, mnist.targets):
            if int(y) in label_dict:
                self.data[label_dict[int(y)]].append(x)

        # shuffle data
        self.length = 0
        random.seed(0)
        for k, v in self.data.items():
            random.shuffle(v)
            self.data[k] = torch.stack(v, dim=0)
            self.length += len(self.data[k])

        # make environments for each env
        # env 0: 0.1, env 1: 0.2, env 3: 0.9
        envs = {
            0: 0.1,
            1: 0.2,
            2: -1,
            3: 0.9,
        }

        if val == 'in_domain':  # use train env for validation
            envs[2] = envs[1]
        else:  # use test env for validation
            envs[2] = envs[3]

        self.envs = {}
        for i in range(4):
            images = []
            labels = []
            for j in range(10):
                start = i*(len(self.data[j])//len(envs))
                end = (i+1)*(len(self.data[j])//len(envs))

                images.append(self.data[j][start:end])
                labels.append((torch.ones(end-start) * j).long())

            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)

            self.envs[i] = self.make_environment(images, labels, envs[i])

        self.data_idx = {
            0: self.envs[0]['idx_dict'],
            1: self.envs[1]['idx_dict'],
            2: self.envs[2]['idx_dict'],
            3: self.envs[3]['idx_dict'],
        }

        # not evaluating worst-case performance of mnist
        self.val_att_idx_dict = None
        self.test_att_idx_dict = None


    def __len__(self):
        return self.length

    @staticmethod
    def make_environment(images, labels, e):
        '''
            https://github.com/facebookresearch/InvariantRiskMinimization
        '''
        # different from the IRM repo, here the labels are already binarized
        images = images.reshape((-1, 28, 28))

        # change label with prob 0.25
        prob_label = torch.ones((10, 10)).float() * (0.25 / 9)
        for i in range(10):
            prob_label[i, i] = 0.75

        labels_prob = torch.index_select(prob_label, dim=0, index=labels)
        labels = Categorical(probs=labels_prob).sample()

        # assign the color variable
        prob_color = torch.ones((10, 10)).float() * (e / 9.0)
        for i in range(10):
            prob_color[i, i] = 1 - e

        color_prob = torch.index_select(prob_color, dim=0, index=labels)
        color = Categorical(probs=color_prob).sample()

        # Apply the color to the image by zeroing out the other color channel
        output_images = torch.zeros((len(images), 10, 28, 28))

        idx_dict = defaultdict(list)
        for i in range(len(images)):
            idx_dict[int(labels[i])].append(i)
            output_images[i, color[i], :, :] = images[i]

        cor = color.float()

        idx_list = list(range(len(images)))

        return {
            'images': (output_images.float() / 255.),
            'labels': labels.long(),
            'idx_dict': idx_dict,
            'idx_list': idx_list,
            'cor': cor,
        }

    def __getitem__(self, keys):
        '''
            @params [support, query]
            support=[(label, y, idx, env)]
            query=[(label, y, idx, env)]
        '''
        idx = []
        if len(keys[0]) == 2:
            # without reindexing y
            idx = []
            for key in keys:
                env_id = int(key[1])
                idx.append(key[0])

            return {
                'X': self.envs[env_id]['images'][idx],
                'Y': self.envs[env_id]['labels'][idx],
                'C': self.envs[env_id]['cor'][idx],
                'idx': torch.tensor(idx).long(),
            }

    def get_all_y(self, env_id):
        return self.envs[env_id]['labels'].tolist()

    def get_all_c(self, env_id):
        return self.envs[env_id]['cor'].tolist()
