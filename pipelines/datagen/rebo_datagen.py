"""
Datagene that takes the ReBO data as input
"""

import os
import torch
import random
import rasterio
import numpy as np
from skimage import io
import kornia.augmentation as K
from kornia.geometry import vflip, hflip
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

random.seed(42)
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

class AlignDatagen:
    def __init__(self, data_dir,
                 sample_size=None,
                 set_name=None,
                 patch_size=None):

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.data_dir = data_dir

        self.set_name = 'train' if set_name == 'val' else set_name
        self.image_dir = os.path.join(data_dir, f'isra_{self.set_name}/')
        self.annotation_path = os.path.join(data_dir, f'ReBO_{self.set_name}.json')
        self.coco = COCO(self.annotation_path)
        self.image_ids = self.coco.getImgIds()

        if set_name == 'train' or set_name == 'val':
            counts = int(0.8 * len(self.image_ids))
            train_ids = set(random.sample(self.image_ids, counts))
            val_ids = list(set(self.image_ids) - train_ids)
            train_ids = list(train_ids)
            self.image_ids = train_ids if set_name == 'train' else val_ids

        self.len = len(self.image_ids)

        # augmentation chances
        self.pixel_noise_chance = 0.15
        self.erasing_chance = 0.15
        self.hflip_chance = 0.5
        self.vflip_chance = 0.5
        self.rot90_chance = 0.5

        # unet agumentation chances
        self.brightness_chance = 0.15
        self.channel_noise_chance = 0.15
        self.pixel_drop_chance = 0.15
        self.pixel_drop_p = 0.05


    def loadSample(self, idx):
        idx = self.image_ids[idx]

        # create an image
        img = self.coco.loadImgs(idx)[0]
        image_path = os.path.join(self.image_dir, img['file_name'])
        image = io.imread(image_path)
        image = image.transpose(2, 0, 1) / 255.0

        # create labels
        annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
        coco_annotations = self.coco.loadAnns(annotation_ids)
        height, width = img['height'], img['width']
        roof_mask = np.zeros((height, width), dtype=np.uint8)
        osm_mask = np.zeros((height, width), dtype=np.uint8)

        for ann in coco_annotations:
            # roof mask
            ann_copy = ann.copy()
            ann_copy['segmentation'] = [ann['roof_mask']]
            rle = self.coco.annToRLE(ann_copy)
            mask = maskUtils.decode(rle)
            roof_mask = np.logical_or(roof_mask, mask).astype(np.uint8)

            # osm mask
            ann_copy = ann.copy()
            ann_copy['segmentation'] = [ann['osm_mask']]
            rle = self.coco.annToRLE(ann_copy)
            mask = maskUtils.decode(rle)
            osm_mask = np.logical_or(osm_mask, mask).astype(np.uint8)

        return image, roof_mask, osm_mask


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, index):
        # generate sample
        image, gold, label = self.loadSample(index)

        # image, label and gold label
        image = torch.from_numpy(image).float()
        gold = torch.from_numpy(gold).float()
        label = torch.from_numpy(label).float()

        return image, gold.unsqueeze(0), label.unsqueeze(0)


    def aug_for_unet(self, X, y, z, device):
        B, C, H, W = X.shape
        y = y.float()

        # flipping horizontal
        hflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.hflip_chance)
        X = X * (1 - hflip_coin) + hflip(X) * hflip_coin
        y = y * (1 - hflip_coin) + hflip(y) * hflip_coin
        z = z * (1 - hflip_coin) + hflip(z) * hflip_coin
        
        # flipping vertical
        vflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.vflip_chance)
        X = X * (1 - vflip_coin) + vflip(X) * vflip_coin
        y = y * (1 - vflip_coin) + vflip(y) * vflip_coin
        z = z * (1 - vflip_coin) + vflip(z) * vflip_coin

        # Rotation 90
        rot90_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        aug = K.RandomRotation90(times=(1, 1), p=1, resample='nearest', keepdim=True)
        X = X * (1 - rot90_coin) + aug(X) * rot90_coin
        y = y * (1 - rot90_coin) + aug(y) * rot90_coin
        z = z * (1 - rot90_coin) + aug(z) * rot90_coin
        
        # Rotation 180
        rot180_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        aug = K.RandomRotation90(times=(2, 2), p=1, resample='nearest', keepdim=True)
        X = X * (1 - rot180_coin) + aug(X) * rot180_coin
        y = y * (1 - rot180_coin) + aug(y) * rot180_coin
        z = z * (1 - rot180_coin) + aug(z) * rot180_coin
        
        # Rotation 270
        rot270_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        aug = K.RandomRotation90(times=(3, 3), p=1, resample='nearest', keepdim=True)
        X = X * (1 - rot270_coin) + aug(X) * rot270_coin
        y = y * (1 - rot270_coin) + aug(y) * rot270_coin
        z = z * (1 - rot270_coin) + aug(z) * rot270_coin

        # Brightness -> per images: Changes brightness between 0.8 and 1.2
        brightness_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.brightness_chance)
        X = X * (1 - brightness_coin) + torch.clip(
            (X + (torch.rand(size=(B, 1, 1, 1), device=device) * 0.4 - 0.2)), 0,
            1) * brightness_coin

        # augmentation from delfors (not using mask)
        # pixelwise noise (each image has a chance of having random noise per pixel)
        # additive noise (uniform and normal/gaussian distribution noise)
        noise_coin = torch.rand((B, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma_n_hyp, sigma_u_hyp = 0.03, 0.3
        sigma = .015 + sigma_n_hyp
        X += ((torch.randn_like(X).clip(-3, 3) * sigma) * noise_coin_n)
        # unoise
        sigma = .05 + sigma_u_hyp
        X += ((torch.rand_like(X) * sigma) * noise_coin_u)

        # multiplicative noise
        noise_coin = torch.rand((B, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()

        # nnoise
        sigma = .005 + sigma_n_hyp
        X += ((X * torch.randn_like(X).clip(-3, 3) * sigma) * noise_coin_n)
        # unoise
        sigma = .015 + sigma_u_hyp
        X += ((X * torch.rand_like(X) * sigma) * noise_coin_u)

        # multiplicative noise
        noise_coin = torch.rand((B, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.channel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.channel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .005 + sigma_n_hyp
        X += ((X * torch.randn((B, C, 1, 1), device=device).clip(-3, 3) * sigma) * noise_coin_n)
        # unoise
        sigma = .015 + sigma_u_hyp
        X += ((X * torch.rand((B, C, 1, 1), device=device) * sigma) * noise_coin_u)

        # pixel dropout
        X *= torch.clip(
            torch.floor(torch.rand((B, C, H, W), device=device) + (1 - self.pixel_drop_p)) +
            torch.floor(torch.rand((B, 1, 1, 1), device=device) + (1 - self.pixel_drop_chance)),
            max=1
        )

        return X, y, z
