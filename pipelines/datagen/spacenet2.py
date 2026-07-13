"""
Datagenerator for
"""
import os
import torch
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import kornia.augmentation as K
from rasterio.features import rasterize
from kornia.geometry import vflip, hflip
from pipelines.utils.matrices import create_affine_matrix, warp_mask_with_affine


class AlignDatagen:
    def __init__(self,
                 data_dir,
                 sample_size=None,
                 dataset_type='synthetic',  # fixme
                 set_name=None,
                 synth_method=50,
                 aug_shift=0,
                 patch_size=None,
                 noise_type="rand",
                 rescale_value=2000):

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.synth_method = synth_method
        self.aug_shift = aug_shift
        self.noise_type = noise_type
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.set_name = set_name
        self.rescale_value = rescale_value
        if self.dataset_type == 'real':
            self.noise_type = 'r'
            file_dir = os.path.join(self.data_dir, 'patch_boundaries_split.geojson')  #fixme
            self.df = gpd.read_file(file_dir)
        else:
            file_dir = os.path.join(self.data_dir, 'data.csv')
            self.df = pd.read_csv(file_dir)
        print(f"Number of images in {self.dataset_type} dataset are: {len(self.df)}")

        if self.set_name is not None:
            self.df = self.df[self.df['split'] == self.set_name].reset_index(drop=True)
            print(f"Number of images in the {self.set_name} set are: {len(self.df)}")
        if sample_size is not None:
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {sample_size} images from the dataset")
        print(f"Final number of images are: {len(self.df)}")

        # augmentation chances
        self.errosion_chance = 0.15
        self.elastic_chance = 0.15
        self.pixel_noise_chance = 0.5
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
        tx_att, ty_att, ro_att = f"Uni_tx_{self.synth_method}", f"Uni_ty_{self.synth_method}", f"Uni_ro_{self.synth_method}"

        if self.noise_type == "u": #uniform
            tx = self.df.iloc[index][tx_att]
            ty = self.df.iloc[index][ty_att]
            theta = self.df.iloc[index][ro_att]/2.0
        elif self.noise_type == "b": #bias
            tx, ty, theta = self.synth_method, 0, 0
        elif self.noise_type == "r":  # real
            pass
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        # pick three random values for augmentation within the range of -aug_shift to aug_shift
        if self.aug_shift is not None and self.aug_shift != 0 and self.noise_type != "r":
            tx += np.random.randint(-self.aug_shift, self.aug_shift+1)
            ty += np.random.randint(-self.aug_shift, self.aug_shift+1)
            aug_shift_rot = round(self.aug_shift/2)
            theta += np.random.randint(-aug_shift_rot, aug_shift_rot)

        image_path = os.path.join(self.data_dir, self.set_name, 'images', self.df.iloc[index]['filename'])
        label_path = os.path.join(self.data_dir, self.set_name, 'labels', self.df.iloc[index]['filename'])

        with rasterio.open(image_path) as src:
            x = src.read().astype(np.float32) / self.rescale_value
        x = torch.from_numpy(x).unsqueeze(0).float()

        with rasterio.open(label_path) as src:
            y = src.read()
        y = torch.from_numpy(y).unsqueeze(0).float()

        assert y.sum().item() != 0, "This datagenerator implemented for only label images"
        if self.noise_type == "r":
            return x.squeeze(0), y.squeeze(0)
        else:
            affine_matrix = create_affine_matrix(tx, ty, theta, self.patch_size)
            warped_mask = warp_mask_with_affine(y, affine_matrix)

            return x.squeeze(0), y.squeeze(0), warped_mask, affine_matrix


    def aug_for_unet(self, X, y, z, device, affine):
        B, C, H, W = X.shape
        y = y.float()
        # five_coins = torch.randint(0, 5, (B, 1, 1, 1), device=device)
        last_row = torch.tensor([0, 0, 1], device=device).unsqueeze(0).repeat(B, 1, 1)
        affine = torch.cat((affine, last_row), dim=1)

        # flipping horizontal
        hflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.hflip_chance)
        # hflip_coin = (five_coins == 0).float()
        X = X * (1 - hflip_coin) + hflip(X) * hflip_coin
        y = y * (1 - hflip_coin) + hflip(y) * hflip_coin
        z = z * (1 - hflip_coin) + hflip(z) * hflip_coin

        hflip_coin = hflip_coin.squeeze(1)
        Fv = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv @ affine @ Fv
        affine = affine * (1 - hflip_coin) + affine_new * hflip_coin
        
        # flipping vertical
        vflip_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.vflip_chance)
        # vflip_coin = (five_coins == 1).float()
        X = X * (1 - vflip_coin) + vflip(X) * vflip_coin
        y = y * (1 - vflip_coin) + vflip(y) * vflip_coin
        z = z * (1 - vflip_coin) + vflip(z) * vflip_coin

        vflip_coin = vflip_coin.squeeze(1)
        Fv = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv @ affine @ Fv
        affine = affine * (1 - vflip_coin) + affine_new * vflip_coin

        # Rotation 90
        rot90_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        # rot90_coin  = (five_coins == 2).float()
        aug = K.RandomRotation90(times=(1, 1), p=1, resample='nearest', keepdim=True)
        X = X * (1 - rot90_coin) + aug(X) * rot90_coin
        y = y * (1 - rot90_coin) + aug(y) * rot90_coin
        z = z * (1 - rot90_coin) + aug(z) * rot90_coin

        rot90_coin = rot90_coin.squeeze(1)
        Fv = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv.transpose(1, 2) @ affine @ Fv
        affine = affine * (1 - rot90_coin) + affine_new * rot90_coin
        
        # Rotation 180
        rot180_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        # rot180_coin = (five_coins == 3).float()
        aug = K.RandomRotation90(times=(2, 2), p=1, resample='nearest', keepdim=True)
        X = X * (1 - rot180_coin) + aug(X) * rot180_coin
        y = y * (1 - rot180_coin) + aug(y) * rot180_coin
        z = z * (1 - rot180_coin) + aug(z) * rot180_coin

        rot180_coin = rot180_coin.squeeze(1)
        Fv = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv.transpose(1, 2) @ affine @ Fv
        affine = affine * (1 - rot180_coin) + affine_new * rot180_coin
        
        # Rotation 270
        rot270_coin = torch.floor(torch.rand((B, 1, 1, 1), device=device) + self.rot90_chance)
        # rot270_coin = (five_coins == 4).float()
        aug = K.RandomRotation90(times=(3, 3), p=1, resample='nearest', keepdim=True)
        X = X * (1 - rot270_coin) + aug(X) * rot270_coin
        y = y * (1 - rot270_coin) + aug(y) * rot270_coin
        z = z * (1 - rot270_coin) + aug(z) * rot270_coin

        rot270_coin = rot270_coin.squeeze(1)
        Fv = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=device).float().unsqueeze(0).repeat(B, 1, 1)
        affine_new = Fv.transpose(1, 2) @ affine @ Fv
        affine = affine * (1 - rot270_coin) + affine_new * rot270_coin

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

        return X, y, z, affine[:, :2, :] if affine is not None else None
