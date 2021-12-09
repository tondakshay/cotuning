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
from loader import main_loading_function 
from loader import Taco


def get_configs():
    """
    Returns a dictionary containing configuration parameters parsed from script arguments.
    """
    parser = argparse.ArgumentParser(
        description="Co-tuning training implementation for EECS 545"
    )

    # train
    parser.add_argument("--gpu", default=0, type=int,
            help="GPU num for training")
    parser.add_argument("--seed", type=int, default=2021)

    # dataset
    parser.add_argument("--classes_num", default=60, type=int,
            help="Number of target domain classes")

    # experiment
    parser.add_argument("--relationship_path",
            default="/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/cotuning/relationship.npy",
            type=str,
            help="Path of pre-computed relationship")

    configs = parser.parse_args()
    return configs


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
    # torch.cuda.set_device(configs.gpu)
    set_seeds(configs.seed)

    # Get loaders for the data for training the final model,
    # relationship learning training, validation data, and test data
    train_loader, val_loader, test_loaders = main_loading_function()

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

        def forward(self, x):
            features = self.feature_net(x)
            out_1 = self.categ_net_1(features)
            out_2 = self.categ_net_2(features)

            return out_1, out_2
    
    net = Net().cuda()    # or just Net() ?
    # Obtain the relationship p(y_s | y_t)
    if os.path.exists(configs.relationship_path):
        print(f"Loading relationship from path: {configs.relationship_path}")
        relationship = np.load(configs.relationship_path)

    else:
        print("Computing the relationsip...")

        os.makedirs(os.path.dirname(configs.relationship_path))
        if os.path.basename(configs.relationship_path) == "":
            configs.relationship_path = os.path.join(configs.relationship_path, "relationship.npy")
        
        def get_features(loader):
            labels_list = []
            logits_list = []

            net.eval()
            for inputs, label in tqdm(loader):
                labels_list.append(label)

                inputs, label = inputs.cuda(), label.cuda()
                source_logits, _ = net(inputs)
                source_logits = source_logits.detach().cpu().numpy()

                logits_list.append(source_logits)
            
            all_logits = np.concatenate(logits_list, axis=0)
            all_labels = np.concatenate(labels_list, axis=0)
            return all_logits, all_labels
        
        rel_train_logits, rel_train_labels = get_features(rel_train_loader)
        rel_val_logits, rel_val_labels = get_features(val_loader)

        relationship = relationship_learning(
            rel_train_logits, rel_train_labels,
            rel_val_logits, rel_val_labels
        )
    
    print(relationship)
    train(configs, train_loader, val_loader, test_loaders, net, relationship)
                
def train(configs, train_loader, val_loader, test_loaders, net, relationship):
    pass

if __name__ == "__main__":
    main()
