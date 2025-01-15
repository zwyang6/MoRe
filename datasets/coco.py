import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import os
import imageio
import sys
from . import transforms
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class_list = ['_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):
    
    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()

def robust_read_image(image_name):
    image = np.asarray(imageio.imread(image_name))
    if len(image.shape)<3:
        image = np.stack((image, image, image), axis=-1)
    return image

class CocoDataset(Dataset):
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
        self.label_dir = os.path.join(root_dir, 'SegmentationClass')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        

        if self.stage == "train":
            img_name = os.path.join(self.img_dir, "train", _img_name+'.jpg')
            if not os.path.exists(img_name):
                img_name = os.path.join(self.img_dir, "val", _img_name+'.jpg')
                image = np.asarray(robust_read_image(img_name))
                label_dir = os.path.join(self.label_dir, "val", _img_name[13:]+'.png')
                # label = np.asarray(imageio.imread(label_dir))
                label = np.asarray(Image.open(label_dir))
            else:
                image = np.asarray(robust_read_image(img_name))
                label_dir = os.path.join(self.label_dir, "train", _img_name[15:]+'.png')
                label = np.asarray(Image.open(label_dir))

        elif self.stage == "val":
            img_name = os.path.join(self.img_dir, "val", _img_name+'.jpg')
            image = np.asarray(robust_read_image(img_name))
            label_dir = os.path.join(self.label_dir, "val", _img_name[13:]+'.png')
            # label = np.asarray(imageio.imread(label_dir))
            label = np.asarray(Image.open(label_dir))


        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label


class CocoClsDataset(CocoDataset):
    def __init__(
        self,
        root_dir,
        name_list_dir,
        split="train",
        stage="train",
        crop_size=448,
        num_classes=81,
        ignore_index=255,
        transform=None,
        aug=True
    ):
        super().__init__(root_dir, name_list_dir, split, stage)
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.transform = transform
        self.random_hflip = T.RandomHorizontalFlip(p=0.5)
        self.random_color_jitter = T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8)
        self.aug = aug
        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)


    def __len__(self):
        return len(self.name_list)

    def transform_image(self, img):
        img = self.random_color_jitter(img)
        img = np.array(img)
        img = transforms.random_scaling(img, scale_range=[0.32, 1.0])
        img = transforms.random_fliplr(img)
        img, img_box = transforms.random_crop(img,crop_size=self.crop_size, mean_rgb=[0, 0, 0], ignore_index=self.ignore_index)
        img = transforms.normalize_img(img)
        # plt.imshow(img)
        # plt.show()
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img, img_box

    def __getitem__(self, idx):

        name, image, label = super().__getitem__(idx)
        cls_label = self.label_list[name]
        pil_image = Image.fromarray(image)

        img, img_box = self.transform_image(pil_image)

        return name, img, cls_label, img_box

class CocoSegDataset(CocoDataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=448,
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
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label
