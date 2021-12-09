from torchvision import datasets
import numpy as np
import os
import json
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


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
def test_transforms(resize_size=256, crop_size=224):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_center = (resize_size - crop_size - 1) / 2

    return transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
    ])



def get_transforms_for_torch(resize_size=256, crop_size=224):
    transforms = {
            'train': training_transforms(resize_size, crop_size),
            'val': validation_transforms(resize_size, crop_size),
            'test' : test_transforms(resize_size, crop_size)
        }

    return transforms

def main_loading_function(dir_path):
    transforms  = get_transforms_for_torch(resize_size = 256, crop_size = 224)

    dataset = Taco()
    coco_obj = dataset.load_taco(dir_path + "/../TACO/data")
    print(dataset.dataset_path)
    #build dataset objects in torch format
    train_dataset = {
        'train':
            datasets.ImageFolder(dataset.dataset_path,
            transform=transforms['train']
        )
    }

    val_dataset = datasets.ImageFolder(
        os.path.join(dataset.dataset_path),
        transform = transforms['val'])

    test_dataset = {
        'test' + str(i):
            datasets.ImageFolder(dataset.dataset_path,
            transform = transforms["test"]
            )
        for i in range(18,25)
    }

    train_loader = DataLoader(train_dataset,batch_size = 10, shuffle = True)
    val_loader = DataLoader(val_dataset,batch_size = 10, shuffle = True)
    test_loader = {
        'test' + str(i):
            DataLoader(
                test_dataset['test' + str(i)],
                batch_size=4, shuffle=False
            )
        for i in range(18,25)
    }

    return train_loader, val_loader, test_loader

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
  #  main_loading_function(dir_path+"../TACO"):
    dataset_train = Taco()
    dataset_train.load_taco(dir_path+"/../TACO/data")


class Taco():
    def __init__(self, class_map=None):
        self.dataset_path = ""
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def load_image(self, image_id):
        """Load the specified image and return as a [H,W,3] Numpy array."""
        # Load image. TODO: do this with opencv to avoid need to correct orientation
        image = Image.open(self.image_info[image_id]['path'])
        img_shape = np.shape(image)
        # load metadata
        exif = image._getexif()
        if exif:
            exif = dict(exif.items())
            # Rotate portrait images if necessary (274 is the orientation tag code)
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)

            # If has an alpha channel, remove it for consistency
        if img_shape[-1] == 4:
            image = image[..., :3]
        return np.array(image)


    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
                "id": image_id,
                "source": source,
                "path": path,
                }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_taco(self, dataset_dir):
        self.dataset_path = dataset_dir 
        ann_filepath = os.path.join(dataset_dir , 'annotations.json')
        dataset = json.load(open(ann_filepath, 'r'))
        taco_alla_coco = COCO()
        taco_alla_coco.dataset = dataset
        taco_alla_coco.createIndex()
        image_ids = []
        class_names = []
        class_ids = sorted(taco_alla_coco.getCatIds())
        for i in class_ids:
            class_name = taco_alla_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                #self.add_class("taco", i, class_name)
                class_names.append(class_name)
                image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
            else:
                background_id = i
        image_ids = list(set(image_ids))

        for i in image_ids:
            self.add_image(
                    "taco", image_id=i,
                    path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
                    width=taco_alla_coco.imgs[i]["width"],
                    height=taco_alla_coco.imgs[i]["height"],
                    annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))
        return taco_alla_coco

class TACO_Dataset(Dataset):
    def __init__(self, annotations_file_path, img_dir, transform=None, target_transform=None):
	with open(annotations_file_path, 'r') as f:
	    annotations_json = json.loads(f.read())




if __name__ == '__main__':
    main()
