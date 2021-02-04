import random

from torch.utils.data import Sampler

from data.mnist import ColoredMNIST
from data.beer import BeerReview
from data.pubmed import Pubmed
from data.celeba import Celeba


class EnvSampler(Sampler):
    def __init__(self, num_batches, batch_size, env_id, idx_list, seed=0):
        '''
            Sample @num_episodes episodes for each epoch. If set to -1, iterate
            through the entire dataet (test mode)

            env_id specifies the env that we are sampling from

            idx_list is the list of data index
        '''
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.env_id = env_id
        self.idx_list = idx_list

        random.seed(seed)

        if self.num_batches == -1:
            self.length = ((len(self.idx_list) + self.batch_size - 1) // self.batch_size)

        else:
            self.length = self.num_batches

    def __iter__(self):
        '''
            Return a list of keys
        '''
        if self.num_batches == -1:
            # iterate through the dataset sequentially
            # for testing
            random.shuffle(self.idx_list)

            # sample through the data
            for i in range(self.length):
                start = i * self.batch_size
                end = min((i+1) * self.batch_size, len(self.idx_list))

                # provide the idx and the env information to the dataset
                yield [(idx, self.env_id) for idx in self.idx_list[start:end]]

        else:
            for _ in range(self.num_batches):
                if self.batch_size < len(self.idx_list):
                    yield [(idx, self.env_id) for idx in
                           random.sample(self.idx_list, self.batch_size)]
                else:
                    # if the number of examples is less than a batch
                    yield [(idx, self.env_id) for idx in self.idx_list]

    def __len__(self):
        return self.length


def get_dataset(data_name, val_type):
    if data_name == 'MNIST':
        train_data = ColoredMNIST('./datasets/mnist', is_train=True,
                                  val=val_type)
        test_data = ColoredMNIST('./datasets/mnist', is_train=False,
                                 val=val_type)

        return train_data, test_data

    if data_name[:4] == 'beer':
        # look: beer_0
        # aroma: beer_1
        # palate: beer_2
        # use the first two env in data for training
        # use the third env in data for validation
        # use the last env in data for testing
        data = BeerReview('./datasets/beer', val=val_type,
                                aspect=data_name[5:])

        return data, data

    if data_name == 'pubmed':
        # use the first two env in data for training
        # use the third env in data for validation
        # use the last env in data for testing
        data = Pubmed('./datasets/pubmed/pubmed.json')

        return data, data

    if data_name == 'celeba':
        # use the first two env in data for training
        # use the third env in data for validation
        # use the last env in data for testing
        data = Celeba('./datasets/celeba')

        return data, data

    raise ValueError('dataset {} is not supported'.format(data_name))
