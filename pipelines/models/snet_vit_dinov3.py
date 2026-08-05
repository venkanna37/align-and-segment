"""
SNet with ConvNeXt-Tiny encoder, DINOv3 weights and custom decoder
"""

import torch
import torch.nn as nn
from .vits import ViT
from .snet_vit import DINOv3Decoder

class Dinov3Seg(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(Dinov3Seg, self).__init__()
        REPO_DIR = '../../../dinov3'
        WEIGHTS_PATH = '../../../dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        self.dinov3 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_PATH)
        self.skip_channels = 64
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, self.skip_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(self.skip_channels),
            nn.GELU(),
            nn.Conv2d(self.skip_channels, self.skip_channels, kernel_size=1, stride=1),
            nn.GELU()
        )
        self.decoder = DINOv3Decoder(384)
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
        encoder_feats = self.dinov3.get_intermediate_layers(x)
        B, N, C = encoder_feats[0].shape  # [1, 400, 384]
        H = W = int(N ** 0.5)  # sqrt(196) = 14
        encoder_feats = encoder_feats[0].permute(0, 2, 1).reshape(B, C, H, W)

        decoder_feats = self.decoder(encoder_feats)
        skip_feats = self.first_block(x)
        decoder_feats = torch.cat([decoder_feats, skip_feats], dim=1)
        decoder_feats = self.last_decoder_block(decoder_feats)
        logits = self.seg_layer(decoder_feats)

        return logits


class ViTSeg(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(ViTSeg, self).__init__()

        self.dinov3 = ViT(num_classes=1, channels=in_channels)
        self.skip_channels = 64
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels, self.skip_channels, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(self.skip_channels),
            nn.GELU(),
            nn.Conv2d(self.skip_channels, self.skip_channels, kernel_size=1, stride=1),
            nn.GELU()
        )
        self.decoder = DINOv3Decoder(384)
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
        encoder_feats = self.dinov3(x)
        B, N, C = encoder_feats.shape  # [1, 400, 384]
        H = W = int(N ** 0.5)  # sqrt(196) = 14
        encoder_feats = encoder_feats.permute(0, 2, 1).reshape(B, C, H, W)

        decoder_feats = self.decoder(encoder_feats)
        skip_feats = self.first_block(x)
        decoder_feats = torch.cat([decoder_feats, skip_feats], dim=1)
        decoder_feats = self.last_decoder_block(decoder_feats)
        logits = self.seg_layer(decoder_feats)

        return logits

