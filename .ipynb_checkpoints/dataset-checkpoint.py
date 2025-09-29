import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import os

import utils

class WildscenesDataset(Dataset):
    def __init__(self, image_path_list, augment):
        self.image_path_list = image_path_list
        for p in self.image_path_list:
            label_path = p.replace("\\image\\", "\\label\\")
            if not os.path.exists(label_path):
                self.image_path_list.remove(p)
                
        self.augment = augment

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        label_path = image_path.replace("\\image\\", "\\label\\")

        with Image.open(image_path) as im:
            im_resized, _ = utils.resize(im, label=False)
            if im_resized.mode != "RGB":
                im_resized = im_resized.convert("RGB")

        with Image.open(label_path) as lb:
            lb_resized, _ = utils.resize(lb, label=True)
            if lb_resized.mode != "RGB":
                lb_resized = lb_resized.convert("RGB")
            
        if self.augment: 
            if random.random() < 0.5:
                im_resized, lb_resized = utils.random_h_flap(im_resized, lb_resized)
                im_resized, lb_resized = utils.random_rotate(im_resized, lb_resized)
                im_resized, lb_resized = utils.random_gaussian_blur(im_resized, lb_resized)
                
        image_tensor = transforms.ToTensor()(im_resized)
        label_masked = utils.rgb_to_idmask(np.array(lb_resized), utils.color2id)
        label_tensor = torch.from_numpy(label_masked).long()

        return image_tensor, label_tensor  