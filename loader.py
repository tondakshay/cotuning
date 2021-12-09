from torchvision import datasets
import numpy as np
import os
import json
from torchvision import transforms
from pycocotools.coco import COCO

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

def main_loading_function():
    transforms  = get_transforms_for_torch(resize_size = 256, crop_size = 224)


    dataset = Taco()
    coco_obj = dataset_train.load_taco(dir_path + "/../TACO/data")
    #build dataset objects in torch format
    train_dataset = {
            'test' + str(i):
            datasets.ImageFolder(os.path.join(dataset.dataset_path, '/batch' + str(i)),
            transform=data_transforms['train'])
            for i in range (17)
        }

    val_dataset = datasets.ImageFolder(
        os.path.join(dataset.data_path, '/batch18'),
        transform=data_transforms['val'])
    test_dataset = {
        'test' + str(i):
            datasets.ImageFolder(os.path.join(path,'test'),
                                 transform = data_transformms["test" + str(i)]
            )
            for i in range(18,25)
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
        self.dataset_path = dataset_dir + "/data"
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

if __name__ == '__main__':
    main()
