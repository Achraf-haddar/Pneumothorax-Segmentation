import os 
import glob
import torch

import numpy as numpy
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

class SIMDataset(Dataset):
    def __init__(self, image_ids, transform=True, preprocessing_fn=None):
        """
        Dataset class for segmentation problem
        :param image_ids: ids of the images, list
        :param transform: True/False, no transform in validation
        :param preprocessing_fn: a function for preprocessing image
        """
        # 




















