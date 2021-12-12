import os
import argparse

import numpy as np
import torch, torchvision
from torch import nn
from tqdm import tqdm

from backbone import ResNet50_F, ResNet50_C
from loader import get_loaders
from main import restore_checkpoint, set_seeds
from logic import softmax

def get_configs():
    parser = argparse.ArgumentParser(
            description="Co-tuning testing"
            )
    parser.add_argument('--load_dir', default="/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/models",
            type=str, help='Directory where models are saved')
    parser.add_argument("--gpu", default=0, type=int,
            help="GPU num for testing")
    parser.add_argument("--classes_num", default=16, type=int,
            help="Number of target domain classes")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=2018, type=int)
    configs = parser.parse_args()
    return configs

def test():
    configs = get_configs()
    if configs.gpu > 0:
        torch.cuda.set_device(configs.gpu)
    set_seeds(configs.seed)

    dir_path = "/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/TACO/data/"
    _, _, _, test_loader = get_loaders(
            dir_path, os.path.join(dir_path, 'annotations.json'),
            batch_size=configs.batch_size
            )

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
    model, _ = restore_checkpoint(net, configs.load_dir)

    logits_list = []
    labels_list = []

    model.eval()
    for image, label in tqdm(test_loader):
        labels_list.append(label)

        if configs.gpu > 0:
            image, label = image.cuda(), label.cuda()
        _, target_logits = model(image)
        target_logits = target_logits.detach().cpu().numpy()

        # print(target_logits.shape)
        logits_list.append(target_logits)

    all_logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    probabilities = softmax(all_logits)
    print("Probabilities shape:", probabilities.shape)
    print("Labels shape:", labels.shape)

    np.save("/scratch/eecs545f21_class_root/eecs545f21_class/atharvp/cotuning/test_probabilities.npy", probabilities)
    np.save("/scratch/eecs545f21_class_root/eecs545f21_class/atharvp/cotuning/test_labels.npy", labels)
    test_accuracy = (probabilities.argmax(axis=1) == labels).mean()
    return test_accuracy

def main():
    test_accuracy = test()
    print(f"Obtained test accuracy = {test_accuracy}")

if __name__ == "__main__":
    main()
