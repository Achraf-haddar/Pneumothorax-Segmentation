import os 
import glob
import torch

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset

from albumentations import(
    Compose, 
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate
)


ImageFile.LOAD_TRUNCATED_IMAGES = True
TRAIN_PATH = "Dataset/train"
MASK_PATH = "Dataset/masks"

class SIIMDataset(Dataset):
    def __init__(self, image_ids, transform=True, preprocessing_fn=None):
        """
        Dataset class for segmentation problem
        :param image_ids: ids of the images, list
        :param transform: True/False, no transform in validation
        :param preprocessing_fn: a function for preprocessing image
        """
        # Create empty dictionary to store image and mask paths
        self.data = defaultdict(dict)
        # for augmentations
        self.transform = transform
        # preprocessing function to normalize images
        self.preprocessing_fn = preprocessing_fn
        # albumentations augmentations
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.1
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )

        # going over all image_ids to store image and mask paths
        counter = 0
        for imgid in image_ids:
            files = glob.glob(os.path.join(TRAIN_PATH, imgid, "*.png"))
            self.data[counter] = {
                "img_path": os.path.join(
                    TRAIN_PATH, imgid + ".png"
                ),
                "mask_path": os.path.join(
                    MASK_PATH, imgid + ".png"
                )
            }
            counter += 1 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        # for a given item index return image and mask tensors
        # read image and mask paths
        img_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]

        # read image and convert to RGB
        img = Image.open(img_path)
        img = img.convert("RGB")

        # PIL image to numpy array
        img = np.array(img)

        # read mask image
        mask = Image.open(mask_path)

        # convert to binary float matrix
        #mask = np.array(mask)  # !!!!!!!!!!!!
        mask = (mask >= 1).astype("float32")
        
        # if this is training data, apply transforms
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        # preprocess the image using provided preprocessing tensors
        # image normalization
        img = self.preprocessing_fn(img)

        # return image and mask tensors
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float()
        }