"""
SNet with ConvNeXt-Tiny encoder, DINOv3 weights and custom decoderer
"""

import timm
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny
from .snet_conv import DINOv3Decoder


class Dinov3(torch.nn.Module):
    def __init__(self, number_of_outputs=4):
        super(Dinov3, self).__init__()
        self.number_of_outputs =number_of_outputs
        self.dinov3 = timm.create_model('convnext_tiny.dinov3_lvd1689m',
                                        pretrained=True,
                                        features_only=True,
                                        out_indices=tuple(range(4 - self.number_of_outputs, 4)))

    def forward(self, x):
        return self.dinov3(x)


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

    def decoder_state_dict(self):
        return {
            "first_block": self.first_block.state_dict(),
            "decoder": self.decoder.state_dict(),
            "last_decoder_block": self.last_decoder_block.state_dict(),
            "seg_layer": self.seg_layer.state_dict(),
        }

    def load_decoder_state_dict(self, state_dict):
        self.first_block.load_state_dict(state_dict["first_block"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.last_decoder_block.load_state_dict(state_dict["last_decoder_block"])
        self.seg_layer.load_state_dict(state_dict["seg_layer"])

    def forward(self, x):
        # use dinov3 features
        encoder_feats = list(self.dinov3(x))
        decoder_feats = self.decoder(encoder_feats)

        skip_feats = self.first_block(x)
        decoder_feats = torch.cat([decoder_feats, skip_feats], dim=1)
        decoder_feats = self.last_decoder_block(decoder_feats)
        logits = self.seg_layer(decoder_feats)

        return logits