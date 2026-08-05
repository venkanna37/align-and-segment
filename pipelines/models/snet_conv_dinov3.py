"""
SNet with ConvNeXt-Tiny encoder, DINOv3 weights and custom decoderer
"""


import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from .snet_conv import DINOv3Decoder
from transformers import AutoModel

class Dinov3(torch.nn.Module):
    def __init__(self, number_of_outputs=4, dinov3_repo_dir=None, weights_path=None):
        super(Dinov3, self).__init__()
        REPO_DIR = dinov3_repo_dir
        WEIGHTS_PATH = weights_path
        # WEIGHTS_PATH = '../../../dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
        self.number_of_outputs = number_of_outputs
        self.dinov3 = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny',
                               source='local',
                               weights=WEIGHTS_PATH)

    def forward(self, x):
        return self.dinov3.get_intermediate_layers(x, n=self.number_of_outputs)


class Dinov3Seg(torch.nn.Module):
    def __init__(self, in_channels=3, dinov3_repo_dir=None, weights_path=None):
        super(Dinov3Seg, self).__init__()

        assert dinov3_repo_dir is not None and weights_path is not None, 'Missing important information'

        self.dinov3 = Dinov3(dinov3_repo_dir=dinov3_repo_dir, weights_path=weights_path)
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