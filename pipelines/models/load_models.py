"""
Load different models and combinations
"""

import torch
from .snet_conv_dinov3 import Dinov3Seg
from .snet_conv import ConvSeg
from .tnet import TNet


# initialize kaiming weights
def init_weights(model):
    for m in model[0].modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)


def load_model(model_name, tnet_backbone=None):

    # Model Name # Encoder        # Weights # Training
    # method1    # ConvNeXt-Tiny  # DINOv3  # Freeze
    # method1a   # ConvNeXt-Tiny  # DINOv3  # Finetune
    # method1b   # ConvNeXt-Tiny  # Random  # Freeze
    # method1c   # ConvNeXt-Tiny  # Random  # Finetune

    # DINOv3 weights in encoder (freeze and finetune)
    if model_name == "method1":
        model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)
        for param in model[0].dinov3.parameters():
            param.requires_grad = False
    elif model_name == "method1a":
        model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder  (freeze and finetune)
    elif model_name == "method1b":
        model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)
        for param in model[0].dinov3.parameters():
            param.requires_grad = False
    else:
        model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    return model