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
# from loader import main_loading_function 
from loader import get_loaders
# from loader import Taco


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

    dir_path = "/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/TACO/data/"
    train_loader, val_loader, test_loaders = get_loaders(dir_path, os.path.join(dir_path, 'annotations.json'))

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
    
    net = Net()#.cuda()    # or just Net() ?
    # Obtain the relationship p(y_s | y_t)
    if os.path.exists(configs.relationship_path):
        print(f"Loading relationship from path: {configs.relationship_path}")
        relationship = np.load(configs.relationship_path)

    else:
        print("Computing the relationship...")

        os.makedirs(os.path.dirname(configs.relationship_path), exist_ok=True)
        if os.path.basename(configs.relationship_path) == "":
            configs.relationship_path = os.path.join(configs.relationship_path, "relationship.npy")
        
        def get_features(ldr):
            labels_list = []
            logits_list = []

            net.eval()
            for inputs, label in tqdm(ldr):
                labels_list.append(label)

                # inputs, label = inputs.cuda(), label.cuda()
                source_logits, _ = net(inputs)
                source_logits = source_logits.detach().cpu().numpy()

                logits_list.append(source_logits)
            
            all_logits = np.concatenate(logits_list, axis=0)
            all_labels = np.concatenate(labels_list, axis=0)
            return all_logits, all_labels
        
        rel_val_logits, rel_val_labels = get_features(val_loader)
        rel_train_logits, rel_train_labels = get_features(train_loader)

        relationship = relationship_learning(
            rel_train_logits, rel_train_labels,
            rel_val_logits, rel_val_labels
        )
        print("Relationship obtained:\n")
        print(relationship)
    
    # train(configs, train_loader, val_loader, test_loaders, net, relationship)
                
def train(configs, train_loader, val_loader, test_loaders, net, relationship):
    total_iters = 5
    train_len = len(train_loader) - 1
    params_list = [{"params": filter(lambda p: p.requires_grad, net.feature_net.parameters())},
                   {"params": filter(lambda p: p.requires_grad,
                                     net.categ_net_1.parameters())},
                   {"params": filter(lambda p: p.requires_grad, net.categ_net_2.parameters()), "lr": 3}] #Setting learning rate 3 for now. SHould be taken from argument parser


    train_iter = iter(train_loader)
    optimizer = torch.optim.SGD(params_list, lr = 3)
  #  scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer)
    for iter_num in tqdm(range(total_iters)):
#Turning the flag on to set the network into training mode
        net.train()
#These are the actual labels against which the loss has to be minimized
        train_inputs, train_labels = next(train_iter)
#pushing data to default GPUs
        imagenet_targets = torch.from_numpy(
            relationship[train_labels]).cuda().float()
        train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
        imagenet_outputs, train_outputs = net(train_inputs)

#running the forward pass
        imagenet_outputs, train_outputs = net(train_inputs)
        
#Loss defined for the whole network is the cross entropy loss. CE loss is generally used when the problem is classification based on 'n' number of classes
        ce_loss = nn.CrossEntropyLoss()(train_outputs, train_labels)


        imagenet_loss = - imagenet_targets * nn.LogSoftmax(dim=-1)(imagenet_outputs)
        loss = ce_loss + configs.trade_off * imagenet_loss

#so the GPU doesn't cry on fast filling memory
        net.zero_grad()
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()
# take a step based on gradient and parameters
        # scheduler.step()




























































if __name__ == "__main__":
    main()
