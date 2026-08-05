"""
Load different models and combinations
"""
import torch
from .tnet import TNet


# initialize kaiming weights
def init_weights(model):
    for m in model[0].modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)


def load_model(model_name, tnet_backbone=None):

    # -------------------ConvNext-Tiny (DINOv3) Model ----------------- #

    # DINOv3 weights in encoder (freeze and finetune)
    if model_name in ['method1', 'method1a']:
        from .snet_conv_dinov3 import Dinov3Seg as Dinov3Seg_Conv
        if model_name == "method1":
            model = Dinov3Seg_Conv(), TNet(in_channels=2, backbone_name=tnet_backbone)
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        else:
            model = Dinov3Seg_Conv(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder  (freeze and finetune)
    if model_name in ['method1b', 'method1c']:
        from .snet_conv import ConvSeg
        if model_name == "method1b":
            model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        else:
            model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # ------------------ ViT-Small Encoder and DINOv3 weights -------- #
    # DINOv3 weights in encoder (freeze and finetune)
    if model_name in ['method2', 'method2a']:
        from .snet_vit_dinov3 import Dinov3Seg
        if model_name == "method2":
            model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        else:
            model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder (freeze and finetune)
    if model_name in ['method2b', 'method2c']:
        from .snet_vit import ViTSeg
        if model_name == "method2b":
            model = ViTSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        else:
            model = ViTSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    return model
