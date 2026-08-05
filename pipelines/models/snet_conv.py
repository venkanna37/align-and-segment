"""
SNets with ConvNext-Tiny as encoder and custom decoder.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny


# SNet with ConvNext-Tiny as encoder
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


class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = convnext_tiny(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.features = self.model.features  # only backbone

    def forward(self, x):
        outputs = []

        x = self.features[0](x)  # patch embed
        x = self.features[1](x)  # stage 1
        outputs.append(x)

        x = self.features[2](x)  # downsample
        x = self.features[3](x)  # stage 2
        outputs.append(x)

        x = self.features[4](x)  # downsample
        x = self.features[5](x)  # stage 3
        outputs.append(x)

        x = self.features[6](x)  # downsample
        x = self.features[7](x)  # stage 4
        outputs.append(x)

        return outputs  # [C1, C2, C3, C4]


class ConvSeg(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(ConvSeg, self).__init__()

        self.dinov3 = ConvNeXtTinyBackbone()
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
        encoder_feats = self.dinov3(x)
        decoder_feats = self.decoder(encoder_feats)

        skip_feats = self.first_block(x)
        decoder_feats = torch.cat([decoder_feats, skip_feats], dim=1)
        decoder_feats = self.last_decoder_block(decoder_feats)
        logits = self.seg_layer(decoder_feats)

        return logits