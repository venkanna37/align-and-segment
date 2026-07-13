import torch
import rasterio
import numpy as np
import pandas as pd
import kornia.augmentation as K
from kornia.geometry import vflip, hflip

import warnings
warnings.filterwarnings('ignore')


class AlignDatagen:
    def __init__(self, data_dir,
                 sample_size=None,
                 set_name=None,
                 patch_size=None,
                 rescale_value=255):

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.set_name = set_name
        self.rescale_value = rescale_value


        file_dir = f"{self.data_dir}/all_rows_splits.csv"
        self.df = pd.read_csv(file_dir)

        if self.set_name is not None:
            self.df = self.df[self.df['split'] == self.set_name].reset_index(drop=True)
            print(f"Number of images in the {self.set_name} set are: {len(self.df)}")

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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.df.shape[0]

    def __getitem__(self, index):
        # Randomly pick one of the images
        with rasterio.open(self.df.iloc[index]['filename']) as src:
            image = src.read().astype(np.float32) / self.rescale_value
        image = torch.from_numpy(image).float()

        with rasterio.open(self.df.iloc[index]['osm_label']) as src:
            label = src.read()
        label = torch.from_numpy(label).float()

        with rasterio.open(self.df.iloc[index]['roof_label']) as src:
            gold = src.read()
        gold = torch.from_numpy(gold).float()

        # random crop
        if self.set_name == 'train' and self.patch_size < 512:
            cx = np.random.randint(0, image.shape[1] - self.patch_size)
            cy = np.random.randint(0, image.shape[2] - self.patch_size)
            image = image[:, cy:cy + self.patch_size, cx:cx + self.patch_size]
            label = label[:, cy:cy + self.patch_size, cx:cx + self.patch_size]
            gold = gold[:, cy:cy + self.patch_size, cx:cx + self.patch_size]

        return image, gold, label

    def create_aug_data(self, y, affine, device):
        B, C, H, W = y.shape

        # affine matrix to (B, 3, 3)
        last_row = torch.tensor([0, 0, 1], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine = torch.cat((affine, last_row), dim=1)

        # horizontal flipping
        hflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.hflip_chance)
        y = y * (1 - hflip_coin) + hflip(y) * hflip_coin
        hflip_coin = hflip_coin.squeeze(1)
        Fv = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv @ affine @ Fv
        affine = affine * (1 - hflip_coin) + affine_new * hflip_coin

        # vertical flipping
        vflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.vflip_chance)
        y = y * (1 - vflip_coin) + vflip(y) * vflip_coin
        vflip_coin = vflip_coin.squeeze(1)
        Fv = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv @ affine @ Fv
        affine = affine * (1 - vflip_coin) + affine_new * vflip_coin

        # rotation 90
        rot90_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        aug = K.RandomRotation90(times=(1, 1), p=1, resample='nearest', keepdim=True)
        y = y * (1 - rot90_coin) + aug(y) * rot90_coin
        rot90_coin = rot90_coin.squeeze(1)
        Fv = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv.transpose(1, 2) @ affine @ Fv
        affine = affine * (1 - rot90_coin) + affine_new * rot90_coin

        # rotation 180
        rot180_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        aug = K.RandomRotation90(times=(2, 2), p=1, resample='nearest', keepdim=True)
        y = y * (1 - rot180_coin) + aug(y) * rot180_coin
        rot180_coin = rot180_coin.squeeze(1)
        Fv = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv.transpose(1, 2) @ affine @ Fv
        affine = affine * (1 - rot180_coin) + affine_new * rot180_coin

        # rotation 270
        rot270_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        aug = K.RandomRotation90(times=(3, 3), p=1, resample='nearest', keepdim=True)
        y = y * (1 - rot270_coin) + aug(y) * rot270_coin
        rot270_coin = rot270_coin.squeeze(1)
        Fv = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv.transpose(1, 2) @ affine @ Fv
        affine = affine * (1 - rot270_coin) + affine_new * rot270_coin

        # additive noise (uniform and normal/gaussian distribution noise)
        noise_coin = torch.rand((B, 1, 1, 1), device=device)
        noise_coin_add = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_rem = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_add).float()
        noise_coin_add = noise_coin_add.float()

        y[:, [0]] += ((torch.rand_like(y[:, [0]]) < 0.02) * noise_coin_add)
        y[:, [0]] -= ((torch.rand_like(y[:, [0]]) < 0.02) * noise_coin_rem)
        # clip to 0-1 range
        y[:, [0]] = torch.clamp(y[:, [0]], 0, 1)

        return y, affine[:, :2, :]


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
