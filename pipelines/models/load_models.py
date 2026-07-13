import torch
from .tnet import TNet
from .dinov3_vit import Dinov3Seg, ViTSeg
from .dinov3_conv import Dinov3Seg as Dinov3Seg_Conv, ConvSeg
from .modified_unet import UNet
from segmentation_models_pytorch import Unet

#fixme add conditional imports when model require pratrained weights
# initialize kaiming weights
def init_weights(model):
    for m in model[0].modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)


def load_model(model_name, tnet_backbone=None, device='cpu',
               in_channels=3, pre_trained_weight=None,
               training=True):
    # ------------------ResNet34 Encoder  ------------------------- #
    # Random weights in encoder and finetune it
    if model_name == "method2":
        model = (UNet(input_channels=in_channels),
                 TNet(in_channels=2, backbone_name=tnet_backbone))
        init_weights(model)

    # Random weights in encoder and freeze it
    elif model_name == "method2a":
        model = (UNet(input_channels=in_channels),
                 TNet(in_channels=2, backbone_name=tnet_backbone))
        init_weights(model)
        for param in model[0].encoder.parameters():
            param.requires_grad = False

    # ImageNet weights in encoder and finetune it
    elif model_name == "method2b":
        model = (Unet(in_channels=3, classes=1, encoder_name='resnet34',
                     encoder_weights='imagenet'),
                 TNet(in_channels=2, backbone_name=tnet_backbone))

    # ImageNet weights in encoder and finetune it
    elif model_name == "method2c":
        model = (Unet(in_channels=3, classes=1, encoder_name='resnet34',
                      encoder_weights='imagenet'),
                 TNet(in_channels=2, backbone_name=tnet_backbone))
        for param in model[0].encoder.parameters():
            param.requires_grad = False

    # ------------------ViT-Small Encoder and DINOv3 weights -------- #
    # DINOv3 weights in encoder and freeze it
    elif model_name == "method3":
        model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)
        # freeze dinov3 weights
        for param in model[0].dinov3.parameters():
            param.requires_grad = False

    # DINOv3 weights in encoder and finetune it
    elif model_name == "method3a":
        model = Dinov3Seg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder and finetune it
    elif model_name == "method3b":
        model = ViTSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder and freeze it
    elif model_name == "method3c": # ViT (small) + freeze
        model = ViTSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)
        for param in model[0].dinov3.parameters():
            param.requires_grad = False

    # -------------------ConvNext-Tiny (DINOv3) Model ----------------- #
    # DINOv3 weights in encoder and freeze it
    elif model_name == "method4":
        model = Dinov3Seg_Conv(), TNet(in_channels=2, backbone_name=tnet_backbone)
        for param in model[0].dinov3.parameters():
            param.requires_grad = False

    # DINOv3 weights in encoder and finetune it
    elif model_name == "method4a":
        model = Dinov3Seg_Conv(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder and finetune it
    elif model_name == "method4b":
        model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)

    # Random weights in encoder and freeze it
    elif model_name == "method4c":
        model = ConvSeg(), TNet(in_channels=2, backbone_name=tnet_backbone)
        for param in model[0].dinov3.parameters():
            param.requires_grad = False

    return model
