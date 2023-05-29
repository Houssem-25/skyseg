import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os
from glob import glob
from utils import pngToCocoResult
class CocoSemantic(Dataset):
    def __init__(self, data_root):
        self.root_dir = os.path.join(data_root,"val2017")
        self.mask_dir = os.path.join(data_root,"panoptic_val2017")
        self.image_files = glob(self.root_dir +"/*")

        self.image_files = sorted(self.image_files , key=lambda x: int(x.split(".")[0].split("/")[-1]))
        self.masks_files = glob(self.mask_dir +"/*")
        self.masks_files = sorted(self.masks_files, key=lambda x: int(x.split(".")[0].split("/")[-1]))
        self.target_sky_color = [75,124,169]
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        mask = np.array(Image.open(self.masks_files[idx]))
        mask = np.all(mask == self.target_sky_color, axis=-1)
        return self.to_tensor(image), self.to_tensor(mask).float()