import numpy as np
import os
import json
import torchvision
from torchvision import transforms, datasets
from torchvision.io import read_image
# from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch import FloatTensor

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
        # return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))
        return transforms.functional.crop(img, self.start_y, self.start_x, th, tw)


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class ToFloatTensor():
    def __call__(self, tensor):
        return tensor.type(FloatTensor)

def training_transforms(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        # ResizeImage(resize_size),
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        ToFloatTensor(),
        normalize
    ])

def validation_transforms(resize_size=256, crop_size=224):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_center = (resize_size - crop_size - 1) // 2

    return transforms.Compose([
        # ResizeImage(resize_size),
        transforms.Resize((resize_size, resize_size)),
        PlaceCrop(crop_size, start_center, start_center),
        # transforms.ToTensor(),
        ToFloatTensor(),
        normalize
    ])
def test_transforms(resize_size=256, crop_size=224):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_center = (resize_size - crop_size - 1) // 2

    return transforms.Compose([
        # ResizeImage(resize_size),
        transforms.Resize((resize_size, resize_size)),
        PlaceCrop(crop_size, start_center, start_center),
        # transforms.ToTensor(),
        ToFloatTensor(),
        normalize
    ])



def get_transforms_for_torch(resize_size=256, crop_size=224):
    transforms = {
            'train': training_transforms(resize_size, crop_size),
            'val': validation_transforms(resize_size, crop_size),
            'test' : test_transforms(resize_size, crop_size)
        }

    return transforms

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    get_loaders(dir_path, dir_path)
    # main_loading_function(dir_path+"../TACO"):
    # dataset_train = Taco()
    # dataset_train.load_taco(dir_path+"/../TACO/data")


def get_loaders(img_dir, ann_path, split=(3800,200,113), random_sampling=True, batch_size=32, limit_size=None):
    # samples = range(4784)
    train_dataset, val_dataset, test_dataset = get_datasets(img_dir, ann_path, split, random_sampling)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    relationship_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, relationship_train_loader, val_loader, test_loader

def get_datasets(img_dir, ann_path, split=(3800,200,113), random_sampling=True, limit_size=None):
    transforms = get_transforms_for_torch(resize_size=256, crop_size=224)
    dataset_size = len(TACO_Dataset(img_dir, ann_path))
    if random_sampling:
        samples = np.random.permutation(dataset_size)    # size of dataset
    else:
        samples = np.arange(dataset_size)

    if limit_size:
        if (limit_size < dataset_size):
            dataset_size = limit_size 

    train_size = split[0] * dataset_size // sum(split)
    val_size = split[1] * dataset_size // sum(split)
    
    train_dataset = TACO_Dataset(img_dir, ann_path, samples[:train_size], transform=transforms['train'])
    val_dataset = TACO_Dataset(img_dir, ann_path, samples[train_size:train_size+val_size], transform=transforms['val'])
    test_dataset = TACO_Dataset(img_dir, ann_path, samples[train_size+val_size:], transform=transforms['test'])
    # train_dataset = TACO_Dataset(img_dir, ann_path, samples[:120], transform=transforms['train'])
    # val_dataset = TACO_Dataset(img_dir, ann_path, samples[120:140], transform=transforms['val'])
    # test_dataset = TACO_Dataset(img_dir, ann_path, samples[140:150], transform=transforms['test'])

    return train_dataset, val_dataset, test_dataset

class TACO_Dataset(Dataset):
    def __init__(self, img_dir, annotations_file_path, samples=None, transform=None, label_transform=None):
        self.img_dir = img_dir

        with open(annotations_file_path, 'r') as f:
            annotations_json = json.loads(f.read())
        image_dict_list = annotations_json['images']
        annotations_dict_list = annotations_json['annotations']
        df1 = pd.DataFrame(image_dict_list, columns=['id', 'width','height', 'file_name'])
        df2 = pd.DataFrame(annotations_dict_list, columns=['image_id', 'category_id', 'bbox'])

        self.df = pd.merge(df1, df2, how='left', left_on='id', right_on='image_id')
        
        supercategories = [
                'Plastic bag & wrapper',
                'Cigarette',
                'Bottle',
                'Bottle cap',
                'Can',
                'Other plastic',
                'Carton',
                'Cup',
                'Straw',
                'Paper',
                'Broken glass',
                'Styrofoam piece',
                'Pop tab',
                'Lid',
                'Plastic container',
                'Aluminium foil',   # select till here
                'Plastic utensils',
                'Rope & strings',
                'Paper bag',
                'Scrap metal',
                'Food waste',
                'Shoe',
                'Squeezable tube',
                'Blister pack',
                'Glass jar',
                'Plastic glooves',
                'Battery',
                'Unlabeled litter',
        ]    # sorted in descending order of data_quantity and relevance

        sc_df = pd.DataFrame(annotations_json['categories'], columns=['supercategory', 'id']).rename(columns={"id":"id2"})
        sc_df['sc_id'] = sc_df['supercategory'].apply(supercategories.index)
        self.df = pd.merge(self.df, sc_df[['id2', 'sc_id']], how='left', left_on='category_id', right_on='id2')
        self.df = self.df.drop(columns=['id', 'id2'])

        self.df = self.df[self.df['sc_id'] < 16].reset_index()    # drop data with fewer training examples

        if samples is not None:
            self.df = self.df.iloc[samples].reset_index()

        self.df['bbox'] = self.df['bbox'].apply(lambda x:list([int(round(z)) for z in x]))
        self.df['file_path'] = self.df['file_name'].apply(lambda file_name: os.path.join(img_dir, file_name))

        self.file_paths = self.df['file_path'].to_numpy()
        self.bboxes = np.array(self.df['bbox'].tolist())
        self.labels = self.df['sc_id'].to_numpy()

        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print(f"Getting item number {idx}...")
        # img_path = os.path.join(self.img_dir, self.df.iloc[idx].loc['file_name'])
        img_path = self.file_paths[idx]
        image = read_image(img_path)

        # bbox = self.df.loc[idx, 'bbox']
        bbox = self.bboxes[idx, :]
        image = torchvision.transforms.functional.crop(
                image,
                left   = bbox[0],
                top    = bbox[1],
                width  = bbox[2],
                height = bbox[3],
        )

        # label = self.df.iloc[idx].loc['sc_id']
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label
    
    def num_examples(self):
        return self.df['sc_id'].value_counts()

# def main_loading_function(dir_path):
#     transforms  = get_transforms_for_torch(resize_size = 256, crop_size = 224)

#     dataset = Taco()
#     coco_obj = dataset.load_taco(dir_path + "/../TACO/data")
#     #build dataset objects in torch format
#     train_dataset = {
#         'train':
#             datasets.ImageFolder(dataset.dataset_path,
#             transform=transforms['train']
#         )
#     }

#     val_dataset = datasets.ImageFolder(
#         os.path.join(dataset.dataset_path),
#         transform = transforms['val'])

#     test_dataset = {
#         'test' + str(i):
#             datasets.ImageFolder(dataset.dataset_path,
#             transform = transforms["test"]
#             )
#         for i in range(18,25)
#     }

#     train_loader = DataLoader(train_dataset,batch_size = 10, shuffle = True)
#     val_loader = DataLoader(val_dataset,batch_size = 10, shuffle = True)
#     test_loader = {
#         'test' + str(i):
#             DataLoader(
#                 test_dataset['test' + str(i)],
#                 batch_size=4, shuffle=False
#             )
#         for i in range(18,25)
#     }

#     return train_loader, val_loader, test_loader

# class Taco():
#     def __init__(self, class_map=None):
#         self.dataset_path = ""
#         self._image_ids = []
#         self.image_info = []
#         # Background is always the first class
#         self.class_info = [{"source": "", "id": 0, "name": "BG"}]
#         self.source_class_ids = {}

    # def load_image(self, image_id):
    #     """Load the specified image and return as a [H,W,3] Numpy array."""
    #     # Load image. TODO: do this with opencv to avoid need to correct orientation
    #     image = Image.open(self.image_info[image_id]['path'])
    #     img_shape = np.shape(image)
    #     # load metadata
    #     exif = image._getexif()
    #     if exif:
    #         exif = dict(exif.items())
    #         # Rotate portrait images if necessary (274 is the orientation tag code)
    #         if 274 in exif:
    #             if exif[274] == 3:
    #                 image = image.rotate(180, expand=True)
    #             if exif[274] == 6:
    #                 image = image.rotate(270, expand=True)
    #             if exif[274] == 8:
    #                 image = image.rotate(90, expand=True)

    #         # If has an alpha channel, remove it for consistency
    #     if img_shape[-1] == 4:
    #         image = image[..., :3]
    #     return np.array(image)


    # def add_image(self, source, image_id, path, **kwargs):
    #     image_info = {
    #             "id": image_id,
    #             "source": source,
    #             "path": path,
    #             }
    #     image_info.update(kwargs)
    #     self.image_info.append(image_info)

    # def load_taco(self, dataset_dir):
    #     self.dataset_path = dataset_dir 
    #     ann_filepath = os.path.join(dataset_dir , 'annotations.json')
    #     dataset = json.load(open(ann_filepath, 'r'))
    #     taco_alla_coco = COCO()
    #     taco_alla_coco.dataset = dataset
    #     taco_alla_coco.createIndex()
    #     image_ids = []
    #     class_names = []
    #     class_ids = sorted(taco_alla_coco.getCatIds())
    #     for i in class_ids:
    #         class_name = taco_alla_coco.loadCats(i)[0]["name"]
    #         if class_name != 'Background':
    #             #self.add_class("taco", i, class_name)
    #             class_names.append(class_name)
    #             image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
    #         else:
    #             background_id = i
    #     image_ids = list(set(image_ids))

    #     for i in image_ids:
    #         self.add_image(
    #                 "taco", image_id=i,
    #                 path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
    #                 width=taco_alla_coco.imgs[i]["width"],
    #                 height=taco_alla_coco.imgs[i]["height"],
    #                 annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
    #                     imgIds=[i], catIds=class_ids, iscrowd=None)))
    #     return taco_alla_coco



if __name__ == '__main__':
    main()
