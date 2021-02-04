import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        resnet = resnet50(pretrained=True, progress=True)
        modules=list(resnet.children())[:-1]
        self.main = nn.Sequential(*modules)
        self.out_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.main(x)

        return x.squeeze(-1).squeeze(-1)
