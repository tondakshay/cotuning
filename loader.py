from torchvision import datasets
import numpy as np
import os
import json
from torchvision import transforms

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def training_transforms(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def validation_transforms(resize_size=256, crop_size=224):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_center = (resize_size - crop_size - 1) / 2

    return transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])

#def test_transforms(resize_size=256, crop_size=224):

def get_transforms_for_torch(resize_size=256, crop_size=224):
    transforms = {
            'train': training_transforms(resize_size, crop_size),
            'val': validation_transforms(resize_size, crop_size),
        }
    transforms.update(test_transforms(resize_size, crop_size))

    return transforms

def main_loading_function(path):
    transforms  = get_transforms_for_torch(resize_size = 256, crop_size = 224)

    #build dataset objects in torch format
    train_dataset = datasets.ImageFolder(
        os.path.join(configs.data_path, 'train'),
        transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(
        os.path.join(configs.data_path, 'val'),
        transform=data_transforms['val'])
    test_dataset = {
        'test' + str(i):
            datasets.ImageFolder(os.path.join(path,'test'),
                                 transform = data_transformms["test" + str(i)]
            )
        for i in range(5)
    }

    train_loader = DataLoader(train_dataset,batch_size = 10, shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size = 10, shuffle = True)
    test_loader = {
        'test' + str(i):
            DataLoader(
                test_datasets["test" + str(i)],
                batch_size=4, shuffle=False, num_workers=configs.num_workers
            )
        for i in range(10)
    }

    return train_loader, val_loader, test_loaders

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
  #  main_loading_function(dir_path+"../TACO"):
    load_taco()

def load_taco(dataset_dir):
    ann_filepath = os.path.join(dataset_dir , 'annotations.json')
    dataset = json.load(open(ann_filepath, 'r'))
    print(dataset)




if __name__ == '__main__':
    main()
