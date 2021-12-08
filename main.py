import os
import argparse
from time import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets 
from tqdm import tqdm

from logic import relationship_learning
from backbone import ResNet50_F, ResNet50_C
from loader import get_loaders


def get_configs():
    """
    Returns a dictionary containing configuration parameters parsed from script arguments.
    """
    return {}  # TODO using argparse


def set_seeds(seed):
    """
    Set seed for random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def main():
    configs = get_configs()
    print(configs)
    torch.cuda.set_device(configs.gpu)
    set_seeds(configs.seed)

    # Get loaders for the data for training the final model,
    # relationship learning training, validation data, and test data
    train_loader, rel_train_loader, val_loader, test_loaders = get_loaders(configs)

    # Define the Neural network class and object which simultaneously predicts source
    # and target domain logits
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_net = ResNet50_F(pretrained=True)
            self.categ_net_1 = ResNet50_C(pretrained=True)
                # outputs source domain logits
            self.categ_net_2 = nn.Linear(self.feature_net.output_dim, configs.classes_num)
                # outputs target domain logits
            torch.nn.init.normal_(self.categ_net_2.weight, 0, 0.01)
            torch.nn.init.constant_(self.categ_net_2.bias, 0.0)

        def forward(x):
            features = self.feature_net(x)
            out_1 = self.categ_net_1(features)
            out_2 = self.categ_net_2(features)

            return out_1, out_2
    
    net = Net().cuda()    # or just Net() ?