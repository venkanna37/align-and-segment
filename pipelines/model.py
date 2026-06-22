"""
DINOv3 segmentation model with convolutional decoder and one skip connection from the input image
Currently supports only DINOv3 ViT-S/16 backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_vit import ViT


def load_model():
    model = Dinov3Seg(), TNet(in_channels=2)
    # freeze dinov3 weights
    for param in model[0].dinov3.parameters():
        param.requires_grad = False

    return model


# TNet model
class TNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.encoder = ViT(channels=in_channels)
        fc_channels = 384

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(fc_channels, 3, bias=False)

    def forward(self, x):
        encoder_features = self.encoder(x)
        encoder_features = encoder_features.mean(dim=1)
        final_values = self.fc(encoder_features)

        params = torch.tanh(final_values) * 0.2
        theta = params[:, 0]
        tx = params[:, 1]
        ty = params[:, 2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        affine_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta, tx], dim=1),
            torch.stack([sin_theta, cos_theta, ty], dim=1)
        ], dim=1)

        return affine_matrix, theta

class DINOv3Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(DINOv3Decoder, self).__init__()

        self.in_channels = in_channels
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels//2),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 4),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 4),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 8, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 8),
            nn.Conv2d(in_channels // 8, in_channels // 8, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 8),
            nn.UpsamplingBilinear2d(scale_factor=2))

        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(in_channels//4, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

    def forward(self, x):
        # x is a list of intermediate features from dinov3
        feats = self.decoder_block1(x[-1])            # 10x10: 768 in channels, 20x20: 384 out channels
        feats = torch.cat((feats, x[-2]), dim=1)      # 20x20: 384+384=768 out channels
        feats = self.decoder_block2(feats)            # 20x20: 768 in channels, 40x40: 192 out channels
        feats = torch.cat((feats, x[-3]), dim=1)      # 40x40: 192+192=384 out channels
        feats = self.decoder_block3(feats)            # 40x40: 384 in channels, 80x80: 96 out channels
        feats = torch.cat((feats, x[-4]), dim=1)      # 80x80: 96+96=192 out channels
        feats = self.decoder_block4(feats)            # 80x80: 192 in channels, 320x320: 48 out channels
        return feats


class Dinov3(torch.nn.Module):
    def __init__(self, number_of_outputs=4):
        super(Dinov3, self).__init__()
        REPO_DIR = '../dinov3'
        WEIGHTS_PATH = '../dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
        self.number_of_outputs = number_of_outputs
        self.dinov3 = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny',
                               source='local',
                               weights=WEIGHTS_PATH)

    def forward(self, x):
        return self.dinov3.get_intermediate_layers(x, n=self.number_of_outputs)


class Dinov3Seg(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(Dinov3Seg, self).__init__()

        self.dinov3 = Dinov3()
        self.skip_channels = 64
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, self.skip_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(self.skip_channels),
            nn.GELU(),
            nn.Conv2d(self.skip_channels, self.skip_channels, kernel_size=1, stride=1),
            nn.GELU()
        )
        self.decoder = DINOv3Decoder(768)
        self.last_decoder_block = nn.Sequential(
            nn.Conv2d(self.skip_channels*2, self.skip_channels*2, 1),
            nn.GELU(),
            nn.BatchNorm2d(self.skip_channels*2),
            nn.Conv2d(self.skip_channels*2, self.skip_channels, 1),
            nn.GELU(),
            nn.BatchNorm2d(self.skip_channels)
        )
        self.seg_layer = nn.Conv2d(self.skip_channels, 1, 1)


    def forward(self, x):
        # use frozen features
        encoder_feats = list(self.dinov3(x))
        for i in range(len(encoder_feats)):
            B, N, C = encoder_feats[i].shape
            H = W = int(N ** 0.5)
            encoder_feats[i] = encoder_feats[i].permute(0, 2, 1).reshape(B, C, H, W)
        decoder_feats = self.decoder(encoder_feats)

        skip_feats = self.first_block(x)
        decoder_feats = torch.cat([decoder_feats, skip_feats], dim=1)
        decoder_feats = self.last_decoder_block(decoder_feats)
        logits = self.seg_layer(decoder_feats)

        return logits


def spatial_transformer_network(input_fmap, theta):
    """
    Spatial Transformer Network in PyTorch using affine_grid and grid_sample.
    Parameters
    ----------
    input_fmap : torch.Tensor
        Input tensor of shape (B, C, H, W)
    theta : torch.Tensor
        Affine transform matrices of shape (B, 2, 3)

    Returns
    -------
    out_fmap : torch.Tensor
        Transformed feature map of shape (B, C, out_H, out_W)
    """
    B, C, H, W = input_fmap.shape

    # Generate the sampling grid
    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=True)

    # Sample the input image with bilinear interpolation
    out_fmap = F.grid_sample(input_fmap, grid, align_corners=True)

    return out_fmap
