import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, include_fc, hidden_dim):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(10, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(9216, hidden_dim)

        self.include_fc = include_fc
        if self.include_fc:
            self.out_dim = hidden_dim
        else:
            self.out_dim = 9216

    def forward(self, input):
        x = input.view(input.shape[0], 10, 28, 28)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        if self.include_fc:
            x = self.fc1(x)
            x = F.relu(x)

        return x
