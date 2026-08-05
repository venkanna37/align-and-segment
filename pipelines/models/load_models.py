"""
Load different models and combinations
"""
import os
import torch
import subprocess
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

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
    # method2    # ViT-Small      # DINOv3  # Freeze
    # method2a   # ViT-Small      # DINOv3  # Finetune
    # method2b   # ViT-Small      # Random  # Freeze
    # method2c   # ViT-Small      # Random  # Finetune

    # DINOv3 weights in encoder
    if model_name in ['method1', 'method1a', 'method2', 'method2a']:
        # image models
        from .snet_conv_dinov3 import Dinov3Seg as Dinov3Seg_Conv
        from .snet_vit_dinov3 import Dinov3Seg

        # Verify dinov3 folder and download if does not exist
        repo_url = "https://github.com/facebookresearch/dinov3.git"
        destination = './dinov3/'

        if not os.path.exists(destination):
            subprocess.run(["git", "clone", repo_url, destination], check=True)
        else:
            print(f"{destination} already exists.")

        # verify pretrained weights and download
        snapshot_download(
            repo_id="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            local_dir=destination)
        safe_weights_path = os.path.join(destination, 'model.safetensors')
        # state_dict = load_file(safe_weights_path)
        # conv_weights_path = os.path.join(destination, 'dinov3_convnext_tiny.pt')
        # torch.save(state_dict, conv_weights_path)

        if model_name == "method1":
            model = (Dinov3Seg_Conv(dinov3_repo_dir=destination, weights_path=safe_weights_path),
                     TNet(in_channels=2, backbone_name=tnet_backbone))
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        elif model_name == "method1a":
            model = (Dinov3Seg_Conv(dinov3_repo_dir=destination, weights_path=conv_weights_path),
                     TNet(in_channels=2, backbone_name=tnet_backbone))

        elif model_name == "method2":
            model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        else:
            model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder  (freeze and finetune)
    if model_name in ['method1b', 'method1c']:
        from .snet_conv import ConvSeg
        if model_name == "method1b":
            model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        else:
            model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    if model_name in ['method2b', 'method2c']:
        from .snet_vit import ViTSeg
        if model_name == "method2b":
            model = ViTSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)
            for param in model[0].dinov3.parameters():
                param.requires_grad = False
        else:
            model = ViTSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    return model
