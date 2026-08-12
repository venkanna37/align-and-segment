"""
SNet with ConvNeXt-Tiny encoder, DINOv3 weights and custom decoderer
"""

import timm
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny


def init_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)


class Dinov3_hub(torch.nn.Module):
    def __init__(self, number_of_outputs=4):
        super(Dinov3_hub, self).__init__()
        REPO_DIR = '../dinov3'
        WEIGHTS_PATH = '../dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
        self.number_of_outputs = number_of_outputs
        self.dinov3 = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny',
                               source='local',
                               weights=WEIGHTS_PATH)

    def forward(self, x):
        encoder_feats = list(self.dinov3.get_intermediate_layers(x, n=self.number_of_outputs))
        for i in range(len(encoder_feats)):
            B, N, C = encoder_feats[i].shape
            H = W = int(N ** 0.5)
            encoder_feats[i] = encoder_feats[i].permute(0, 2, 1).reshape(B, C, H, W)
        return encoder_feats


class Dinov3_timm(torch.nn.Module):
    def __init__(self, number_of_outputs=4):
        super(Dinov3_timm, self).__init__()
        self.number_of_outputs =number_of_outputs
        self.dinov3 = timm.create_model('convnext_tiny.dinov3_lvd1689m',
                                        pretrained=True,
                                        features_only=True,
                                        out_indices=tuple(range(4 - self.number_of_outputs, 4)))

    def forward(self, x):
        return self.dinov3(x)


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


class Dinov3Seg(torch.nn.Module):
    def __init__(self, in_channels=3, method='method1'):
        super(Dinov3Seg, self).__init__()

        if method == 'method1':
            self.dinov3 = Dinov3_hub()
        elif method == 'method2':
            self.dinov3 = Dinov3_timm()
        else:
            raise NotImplementedError

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

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for decoder components."""
        for module in [self.first_block, self.decoder, self.last_decoder_block, self.seg_layer]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

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