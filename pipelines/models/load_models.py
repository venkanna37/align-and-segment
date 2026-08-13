"""
Load different models and combinations
"""

import torch
from .snet import Dinov3Seg
from .tnet import TNet


def load_model(model_name, tnet_backbone=None, tnet_scale_factor=0.35):

    model = Dinov3Seg(method=model_name), TNet(in_channels=2, backbone_name=tnet_backbone,
                                               tnet_scale_factor=tnet_scale_factor)
    for param in model[0].dinov3.parameters():
        param.requires_grad = False

    return model