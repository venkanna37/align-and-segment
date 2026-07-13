import torch
import torch.nn as nn

from .resnets import ResNet11, ResNet18, ResNet34
from .vits import ViT


class TNet(nn.Module):
    def __init__(self, in_channels=2, backbone_name='vitsmall'):
        super().__init__()
        self.encoder_channels = [64, 64, 128, 256, 512]
        fc_channels = self.encoder_channels[-1]
        self.dim_size = 384
        self.backbone_name = backbone_name
        if self.backbone_name == 'resnet11':
            self.encoder = ResNet11(in_channels, self.encoder_channels)
        elif self.backbone_name == 'resnet18':
            self.encoder = ResNet18(in_channels, self.encoder_channels)
        elif self.backbone_name == 'resnet34':
            self.encoder = ResNet34(in_channels, self.encoder_channels)
        elif self.backbone_name == 'vitsmall':
            self.encoder = ViT(channels=in_channels)
            fc_channels = self.dim_size
        elif self.backbone_name == 'vitmedium':
            self.dim_size = 256
            fc_channels = self.dim_size
            self.encoder = ViT(heads=3, dim=self.dim_size, mlp_dim=1152)
        elif self.backbone_name == 'vittiny':
            self.dim_size = 192
            fc_channels = self.dim_size
            self.encoder = ViT(heads=3, dim=self.dim_size, mlp_dim=768)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(fc_channels, 3, bias=False)

    def forward(self, x):
        encoder_features = self.encoder(x)
        if "vit" in self.backbone_name:
            encoder_features = encoder_features.mean(dim=1)
            final_values = self.fc(encoder_features)
        else:
            final_features = encoder_features[-1]
            x = self.pool(final_features)
            x = x.view(final_features.size(0), -1)
            final_values = self.fc(x)

        params = torch.tanh(final_values) * 0.35  # predicts misalignment range from -128 to 128 pixels
        theta = params[:, 0]
        tx = params[:, 1]
        ty = params[:, 2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        affine_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta, tx], dim=1),
            torch.stack([sin_theta, cos_theta, ty], dim=1)
        ], dim=1)

        return affine_matrix, theta