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

def restore_checkpoint(model, save_dir):
    files = [files for files in os.listdir(save_dir) if files.endswith('.pkl')]
    start_iter = len(files)
    if not files:
        print("No saved files found")
        return model, 0
    try:
        filename = os.path.join(save_dir,f'{start_iter-1}.pkl')
        print("Loading file", filename)
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'], strict = False)
        print("Loaded from checkpoint successfully from {}".format(filename))
    except:
        print("No model found")
    return model, start_iter 

def get_configs():
    """
    Returns a dictionary containing configuration parameters parsed from script arguments.
    """
    parser = argparse.ArgumentParser(
        description="Co-tuning training implementation for EECS 545"
    )

    # in
    parser.add_argument("--gpu", default=0, type=int,
            help="GPU num for training")
    

    parser.add_argument("--seed", default=2018, type=int)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument('--total_iters', default=9050, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for training')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma value for learning rate decay')

    # dataset
    parser.add_argument("--split", nargs='+', type=int, default=[3800, 200, 113],
            help="Split proportion of train, val, test")
    parser.add_argument("-lsize", "--limit-size", type=int, default=4113,
            help="Select limited portion of the dataset for train, val, test")
    parser.add_argument("--classes_num", default=60, type=int,
            help="Number of target domain classes")

    # experiment
    parser.add_argument("-reldir", "--relationship_dir",
            default="/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/cotuning/",
            type=str,
            help="Path of pre-computed relationship")
    parser.add_argument('--save-dir', default="/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/models",
                        type=str, help='Path of saved models')
    parser.add_argument("-logitdir", "--logits-dir",
            default="/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/cotuning/logits/",
            help="Path of logits storage")
    parser.add_argument('-fr', "--force-recompute", nargs='+', default=[],
            help="Forcibly recompute these aspects")

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
    if configs.gpu > 0:
        torch.cuda.set_device(configs.gpu)
    set_seeds(configs.seed)

    # Get loaders for the data for training the final model,
    # relationship learning training, validation data, and test data

    dir_path = "/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/TACO/data/"
    train_loader, rel_train_loader, val_loader, test_loaders = get_loaders(
            dir_path, os.path.join(dir_path, 'annotations.json'),
            split=configs.split, limit_size=configs.limit_size)

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
    
    net = Net()
    if configs.gpu > 0:
        net = net.cuda()

    # Obtain the relationship p(y_s | y_t)
    rel_path = os.path.join(configs.relationship_dir, f'rel_{configs.seed}.npy')
    if os.path.exists(rel_path) and ('rel' not in configs.force_recompute):
        print(f"Loading relationship from path: {configs.relationship_dir}")
        relationship = np.load(rel_path)

    else:
        print("Need to compute the relationship...")

        os.makedirs(configs.relationship_dir, exist_ok=True)
        # if os.path.basename(configs.relationship_path) == "":
            # configs.relationship_path = os.path.join(configs.relationship_path, "relationship.npy")
        def get_features(ldr):
            labels_list = []
            logits_list = []

            net.eval()
            for inputs, label in tqdm(ldr):
                labels_list.append(label)

                if configs.gpu > 0:
                    inputs, label = inputs.cuda(), label.cuda()
                source_logits, _ = net(inputs)
                source_logits = source_logits.detach().cpu().numpy()

                logits_list.append(source_logits)
            
            all_logits = np.concatenate(logits_list, axis=0)
            all_labels = np.concatenate(labels_list, axis=0)
            return all_logits, all_labels
        
        relt_logits_path = os.path.join(configs.logits_dir, f'relt_logits_{configs.limit_size}.npy')
        relt_labels_path = os.path.join(configs.logits_dir, f'relt_labels_{configs.limit_size}.npy')
        relv_logits_path = os.path.join(configs.logits_dir, f'relv_logits_{configs.limit_size}.npy')
        relv_labels_path = os.path.join(configs.logits_dir, f'relv_labels_{configs.limit_size}.npy')
        os.makedirs(configs.logits_dir, exist_ok=True)

        if ('logits' not in configs.force_recompute):
            try:
                rel_val_logits, rel_val_labels = np.load(relv_logits_path), np.load(relv_labels_path)
                rel_train_logits, rel_train_labels = np.load(relt_logits_path), np.load(relt_labels_path)
            except FileNotFoundError:
                print("Logits not found. Computing logits for relationship training...")
                rel_val_logits, rel_val_labels = get_features(val_loader)
                rel_train_logits, rel_train_labels = get_features(rel_train_loader)
                np.save(relt_logits_path, rel_train_logits)
                np.save(relt_labels_path, rel_train_labels)
                np.save(relv_logits_path, rel_val_logits)
                np.save(relv_labels_path, rel_val_labels)
        else:
            print("Computing logits for relationship training...")
            rel_val_logits, rel_val_labels = get_features(val_loader)
            rel_train_logits, rel_train_labels = get_features(rel_train_loader)
            np.save(relt_logits_path, rel_train_logits)
            np.save(relt_labels_path, rel_train_labels)
            np.save(relv_logits_path, rel_val_logits)
            np.save(relv_labels_path, rel_val_labels)

        print("Now computing relationship...")
        relationship = relationship_learning(
            rel_train_logits, rel_train_labels,
            rel_val_logits, rel_val_labels
        )
        print("Relationship obtained:")
        print(relationship, "\n")
        print("Relationship shape: ", relationship.shape, "\n")
        np.save(rel_path, relationship)

    train(configs, train_loader, val_loader, test_loaders, net, relationship)
                
def train(configs, train_loader, val_loader, test_loaders, net, relationship):


    train_len = len(train_loader) - 1
    params_list = [{"params": filter(lambda p: p.requires_grad, net.feature_net.parameters())},
                   {"params": filter(lambda p: p.requires_grad,
                                     net.categ_net_1.parameters())},
                   {"params": filter(lambda p: p.requires_grad, net.categ_net_2.parameters()), "lr": 3}] #Setting learning rate 3 for now. SHould be taken from argument parser

    train_iter = iter(train_loader)
    optimizer = torch.optim.SGD(params_list, lr = 3)
    milestones = [6000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=0.1)
    net,start_iter = restore_checkpoint(net, configs.save_dir)
    for iter_num in tqdm(range(start_iter, configs.total_iters)):
#Turning the flag on to set the network into training mode
        net.train()
        if iter_num % train_len == 0:
            train_iter = iter(train_loader)

#These are the actual labels against which the loss has to be minimized
        train_inputs, train_labels = next(train_iter)
#pushing data to default GPUs
        if configs.gpu > 0:
            imagenet_targets = torch.from_numpy(
                    relationship[train_labels]).cuda().float()
            train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
        else:
            imagenet_targets = torch.from_numpy(
                relationship[train_labels]).float()

#running the forward pass
        imagenet_outputs, train_outputs = net(train_inputs)
        
#Loss defined for the whole network is the cross entropy loss. CE loss is generally used when the problem is classification based on 'n' number of classes
        ce_loss = nn.CrossEntropyLoss()(train_outputs, train_labels)


        imagenet_loss = - imagenet_targets * nn.LogSoftmax(dim=-1)(imagenet_outputs)
        imagenet_loss = torch.mean(torch.sum(imagenet_loss, dim=-1))
        loss = ce_loss + 2.3 * imagenet_loss

#so the GPU doesn't cry on fast filling memory
        net.zero_grad()
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()
# take a step based on gradient and parameters
# scheduler.step()
        scheduler.step()
        # print(
        #   "Iter: {}/{} ".format(
        #       iter_num, 9050)
        #   )
        checkpoint = {
                'state_dict': net.state_dict(),
                'iter': iter_num,
            }
        torch.save(checkpoint,
                       os.path.join(configs.save_dir, '{}.pkl'.format(iter_num)))
        # print("Model Saved.")









if __name__ == "__main__":
    main()
