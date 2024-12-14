import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
from PIL import Image
from torchvision import transforms as T


class_list = ["_background_", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):
    
    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()

class VOC12Dataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label


class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 num_classes=21,
                 transform=None,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.num_classes = num_classes

        self.transform = transform

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)
        pil_image = Image.fromarray(image)
        cls_label = self.label_list[img_name]

        images = self.transform(pil_image)
        input = images[1]

        return img_name, input, cls_label, images


class VOC12SegDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = transforms.random_crop(image, label, crop_size=self.crop_size, mean_rgb=[123.675, 116.28, 103.53], ignore_index=self.ignore_index)
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)
        
        if self.stage == "test":
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        return img_name, image, label,  cls_label
