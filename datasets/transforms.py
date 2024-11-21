import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomResizedCrop:
    def __init__(self, img_size, img_scale):
        self.img_size = img_size
        self.img_scale = img_scale
    
    def __call__(self, sample):
        crop_params = T.RandomResizedCrop(self.img_size).get_params(sample['image'], self.img_scale, [3/4, 4/3])
        sample['image'] = F.resized_crop(sample['image'], *crop_params, [self.img_size, self.img_size], InterpolationMode.BILINEAR)
        sample['mask'] = F.resized_crop(sample['mask'], *crop_params, [self.img_size, self.img_size], InterpolationMode.NEAREST)
        return sample


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        if random.random() < self.flip_prob:
            sample['image'] = F.hflip(sample['image'])
            sample['mask'] = F.hflip(sample['mask'])
        return sample


class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        sample['image'] = F.to_tensor(sample['image'])
        sample['mask'] = torch.from_numpy(sample['mask']).contiguous()
        return sample


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample

class ColorJitter:  
    def __init__(self, prob=0.4, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):  
        # 初始化ColorJitter  
        self.color_jitter = T.ColorJitter(brightness=brightness,  
                                                contrast=contrast,  
                                                saturation=saturation,  
                                                hue=hue) 
        self.prob = prob 
    def __call__(self, sample):  
        # 应用ColorJitter  
        if random.random() < self.prob:
            sample['image'] = self.color_jitter(sample['image'])  
          
        return sample 