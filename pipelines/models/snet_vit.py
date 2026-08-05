"""
SNets with ViT as encoder and custom decoder.
"""

import torch
import torch.nn as nn
from .vits import ViT

# SNet with ViT as encoder
class DINOv3Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(DINOv3Decoder, self).__init__()

        self.in_channels = in_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 2),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 4),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 4),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels // 4, in_channels // 6, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 6),
            nn.Conv2d(in_channels // 6, in_channels // 6, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(in_channels // 6),

            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        return self.decoder(x)


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
            nn.Conv2d(self.skip_channels * 2, self.skip_channels * 2, 1),
            nn.GELU(),
            nn.BatchNorm2d(self.skip_channels * 2),
            nn.Conv2d(self.skip_channels * 2, self.skip_channels, 1),
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

