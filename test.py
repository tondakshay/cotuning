import os
import argparse

import numpy as np
import torch, torchvision
from tqdm import tqdm

from backbone import ResNet50_F, ResNet50_C
from loader import get_loader
from main import restore_checkpoint
from logic import softmax

def get_configs():
    parser = arparse.ArgumentParser(
            description="Co-tuning testing"
            )
    parser.add_argument('--load_dir', default="/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/models",
            type=str, help='Directory where models are saved')
    parser.add_argument("--gpu", default=0, type=int,
            help="GPU num for testing")
    configs = parser.parse_args()
    return configs

def test():
    dir_path = "/scratch/eecs545f21_class_root/eecs545f21_class/akshayt/TACO/data/"
    _, _, _, test_loader = get_loaders(
            dir_path, os.path.join(dir_path, 'annotations.json'),
            split=configs.split, limit_size=configs.limit_size)

    model, _ = restore_checkpoint(None, configs.load_dir)

    test_iter = iter(test_loader)
    logits_list = []
    labels_list = []

    model.eval()
    for image, label in tqdm(test_loader):
        labels_list.append(label)

        if configs.gpu > 0:
            image, label = image.cuda(), label.cuda()
        _, target_logits = model(image)
        target_logits = target_logits.detach().cpu().numpy()

        logits_list.append(target_logits)
   all_logits = np.concatenate(logits_list, axis=0)
   labels = np.concatenate(labels_list, axis=0)

   probabilities = softmax(all_logits)

   test_accuracy = (probabilities.argmax(axis=1) == labels).mean()
   return test_accuracy

def main():
    test_accuracy = test()
    print(f"Obtained test accuracy = {test_accuracy}")

if __name__ == "__main__":
    main()
